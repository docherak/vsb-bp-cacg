#include <petscksp.h>

static char help[] = "Solves a linear system by loading a matrix from a file and using a default RHS vector.\n\n";

int main(int argc, char **args)
{
  Mat            A;
  Vec            b;
  PC             pc;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscLogDouble t1, t2;
  PetscViewer    viewer;
  char           file[PETSC_MAX_PATH_LEN];
  PetscScalar    tol = 1e-6; // Default tolerance

  ierr = PetscInitialize(&argc, &args, (char *)0, help);
  CHKERRQ(ierr);

  /* --- Get user-specified options --- */
  ierr = PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg);
  CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate matrix file with -f <filename>");
  ierr = PetscOptionsGetScalar(NULL, NULL, "-tol", &tol, &flg);
  CHKERRQ(ierr);

  /* --- Load ONLY the matrix from the file --- */
  PetscPrintf(PETSC_COMM_WORLD, "Loading matrix from file: %s\n", file);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer);
  CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);
  CHKERRQ(ierr);
  ierr = MatLoad(A, viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);
  CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_SPD, PETSC_TRUE);

  /* --- Create a default right-hand-side vector 'b' of all ones --- */
  PetscPrintf(PETSC_COMM_WORLD, "Creating a default RHS vector 'b' of all ones.\n");
  ierr = MatCreateVecs(A, NULL, &b);
  CHKERRQ(ierr);
  ierr = VecSet(b, 1.0);
  CHKERRQ(ierr);

  /* --- Setup Preconditioner (PC) --- */
  ierr = PCCreate(PETSC_COMM_WORLD, &pc);
  CHKERRQ(ierr);
  ierr = PCSetOperators(pc, A, A);
  CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); // Allows user to choose PC type via command line (e.g., -pc_type jacobi)
  CHKERRQ(ierr);
  ierr = PCSetUp(pc);
  CHKERRQ(ierr);

  /* --- CG Solver --- */
  ierr = PetscTime(&t1);
  CHKERRQ(ierr);

  Vec         x_k, r_k, u_k, w_k, d_k, s_k;
  PetscScalar gamma_k, gamma_old, delta_k, alpha_k, beta_k, r_norm;
  PetscInt    max_iter = 100000, iter;
  PetscInt    rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = VecDuplicate(b, &x_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &r_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &u_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &w_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &d_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &s_k);
  CHKERRQ(ierr);

  ierr = VecSet(x_k, 0.0);
  CHKERRQ(ierr); // paper: x_0
  ierr = VecCopy(b, r_k);
  CHKERRQ(ierr); // paper: r_0 = b - Ax_0 = b - 0 = b
  ierr = PCApply(pc, r_k, u_k);
  CHKERRQ(ierr); // paper: u_0 = M^(-1)r_0
  ierr = MatMult(A, u_k, w_k);
  CHKERRQ(ierr); // paper: w_0 = Au_0

  ierr = VecDot(r_k, u_k, &gamma_k);
  CHKERRQ(ierr); // paper: (r_0, u_0)
  ierr = VecDot(w_k, u_k, &delta_k);
  CHKERRQ(ierr);               // paper: (w_0, u_0)
  alpha_k = gamma_k / delta_k; // paper: (r_0, u_0) / (w_0, u_0)
  beta_k  = 0;

  // initialize p_0 = u_0 and s_0 = w_0 (since beta=0)
  ierr = VecCopy(u_k, d_k);
  CHKERRQ(ierr);
  ierr = VecCopy(w_k, s_k);
  CHKERRQ(ierr);

  PetscScalar natural_norm = PetscSqrtScalar(PetscAbsScalar(gamma_k));

  if (rank == 0) { PetscPrintf(PETSC_COMM_WORLD, "\nStarting PCG (Alg 2) with tolerance %g\n", (double)tol); }

  for (iter = 0; iter < max_iter; iter++) {
    if (natural_norm < tol) { break; }

    if (iter > 0) {
      ierr = VecScale(d_k, beta_k);  // p = beta * p_old
      ierr = VecAXPY(d_k, 1.0, u_k); // p = p + u

      ierr = VecScale(s_k, beta_k);  // s = beta * s_old
      ierr = VecAXPY(s_k, 1.0, w_k); // s = s + w
    }

    ierr = VecAXPY(x_k, alpha_k, d_k);  // x = x + alpha * p
    ierr = VecAXPY(r_k, -alpha_k, s_k); // r = r - alpha * s

    ierr = PCApply(pc, r_k, u_k); // u_{i+1} = M^-1 r_{i+1}
    ierr = MatMult(A, u_k, w_k);  // w_{i+1} = A u_{i+1}

    gamma_old = gamma_k;
    ierr      = VecDot(r_k, u_k, &gamma_k);
    CHKERRQ(ierr);
    ierr = VecDot(w_k, u_k, &delta_k);
    CHKERRQ(ierr);

    natural_norm = PetscSqrtScalar(PetscAbsScalar(gamma_k));

    if (gamma_old == 0.0) {
      beta_k = 0.0;
    } else {
      beta_k = gamma_k / gamma_old;
    }

    PetscScalar denominator = delta_k - ((beta_k / alpha_k) * gamma_k);
    if (denominator == 0.0) {
      PetscPrintf(PETSC_COMM_WORLD, "Breakdown: Denominator is zero.\n");
      break;
    }
    alpha_k = gamma_k / denominator;
  }

  ierr = PetscTime(&t2);
  CHKERRQ(ierr);
  ierr = VecNorm(r_k, NORM_2, &r_norm);
  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "\n--- Final Solution Vector (x_k) ---\n");
  ierr = VecView(x_k, PETSC_VIEWER_STDOUT_WORLD);
  if (rank == 0) {
    if (natural_norm <= tol) { // Use natural_norm here
      PetscPrintf(PETSC_COMM_WORLD, "PCG converged on Natural Norm in %d iterations.\n", (int)iter);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "PCG did NOT converge. Final Natural Norm: %g\n", (double)natural_norm);
    }
  }
  PetscPrintf(PETSC_COMM_WORLD, "Time of solution : %5.3e\n", t2 - t1);
  CHKERRQ(ierr);

  /* --- Cleanup --- */
  ierr = MatDestroy(&A);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = VecDestroy(&x_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&r_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&d_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&s_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&u_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&w_k);
  CHKERRQ(ierr);
  ierr = PCDestroy(&pc);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
