#include "petscsys.h"
#include <petscksp.h>

static char help[] = "Solves a linear system by loading a matrix from a file and using a default RHS vector.\n\n";

int main(int argc, char **args)
{
  Mat            A;
  Vec            b;
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

  /* --- CG Solver --- */
  ierr = PetscTime(&t1);
  CHKERRQ(ierr);

  Vec         x_k, r_k, p_k, q_k;
  PetscScalar r_norm, r0_norm, r_k_dot_r_k, r_k_pre_dot_r_k_pre, r_k_dot_q_k, alpha_k, beta_k;
  PetscInt    max_iter = 100000, iter;
  PetscInt    rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = VecDuplicate(b, &x_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &r_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &p_k);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &q_k);
  CHKERRQ(ierr);

  ierr = VecSet(x_k, 0.0);
  CHKERRQ(ierr); // paper: x_0
  ierr = VecCopy(b, r_k);
  CHKERRQ(ierr); // paper: r_0 = f - Ax_0 = b - Ax_0 = b - 0 = b

  ierr = VecNorm(r_k, NORM_2, &r_norm);
  CHKERRQ(ierr); // paper: Convergence criteria

  ierr = VecNorm(r_k, NORM_2, &r0_norm);
  CHKERRQ(ierr);

  if (r0_norm == 0.0) r0_norm = 1.0;

  if (rank == 0) { PetscPrintf(PETSC_COMM_WORLD, "\nStarting CG solver with tolerance %g\n", (double)tol); }

  for (iter = 0; iter < max_iter; iter++) {
    if (iter == 0) {
      ierr = VecCopy(r_k, p_k);
      CHKERRQ(ierr); // paper: p_k = r_k
      ierr = VecDot(r_k, r_k, &r_k_dot_r_k);
      CHKERRQ(ierr);
    } else {
      // paper: Compute Beta_i
      r_k_pre_dot_r_k_pre = r_k_dot_r_k;
      ierr                = VecDot(r_k, r_k, &r_k_dot_r_k);
      CHKERRQ(ierr);
      beta_k = r_k_dot_r_k / r_k_pre_dot_r_k_pre;
      // paper: Compute p_i = r_i + (Beta_i, p_(i-1))
      ierr = VecScale(p_k, beta_k);
      CHKERRQ(ierr);
      ierr = VecAXPY(p_k, 1.0, r_k);
      CHKERRQ(ierr);
    }

    // paper: Compute q_i = Ap_i
    ierr = MatMult(A, p_k, q_k);
    CHKERRQ(ierr);

    // paper: Compute Alpha_i = (r_i, r_i) / (r_i, q_i)
    ierr = VecDot(r_k, q_k, &r_k_dot_q_k);
    CHKERRQ(ierr);
    alpha_k = r_k_dot_r_k / r_k_dot_q_k;

    // paper: Compute x_(i+1)
    ierr = VecAXPY(x_k, alpha_k, p_k);
    CHKERRQ(ierr);

    // paper: Compute r_(i+1)
    ierr = VecAXPY(r_k, -alpha_k, q_k);
    CHKERRQ(ierr);

    ierr = VecNorm(r_k, NORM_2, &r_norm);
    CHKERRQ(ierr);
    if (r_norm / r0_norm < tol) break;
  }

  ierr = PetscTime(&t2);
  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "\n--- Final Solution Vector (x_k) ---\n");
  ierr = VecView(x_k, PETSC_VIEWER_STDOUT_WORLD);
  if (rank == 0) {
    if (r_norm / r0_norm <= tol) {
      PetscPrintf(PETSC_COMM_WORLD, "CG converged in %D iterations.\n", iter);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "CG did NOT converge in %D iterations. Final residual norm: %g\n", iter, (double)r_norm);
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
  ierr = VecDestroy(&q_k);
  CHKERRQ(ierr);
  ierr = VecDestroy(&p_k);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
