#include "petscsystypes.h"
#include <petscksp.h>
#include <petscblaslapack.h>

static char help[] = "Solves linear system using s-step CG (CACG) with Monomial Basis (Fixed).\n"
                     "Implements Algorithm from Carson et al. (2022), Figure 2 (Uniform Precision).\n\n";

// Helper to apply shift matrix T for monomial basis on CPU vectors
// Represents multiplication by A in the coordinate basis: v_i to v_{i+1}
// dest = T * src
void ApplyMonomialT(PetscInt s, const PetscScalar *src, PetscScalar *dest)
{
  PetscInt dim = 2 * s + 1;
  PetscInt i;

  for (i = 0; i < dim; i++) dest[i] = 0.0;

  // Shift P part: indices 0 to s-1 map to 1 to s
  // Corresponds to P_{j+1} = A * P_j in basis generation
  for (i = 0; i < s; i++) { dest[i + 1] += src[i]; }

  // Shift R part: indices s+1 to 2s-1 map to s+2 to 2s
  // Corresponds to R_{j+1} = A * R_j
  for (i = 0; i < s - 1; i++) { dest[s + 1 + i + 1] += src[s + 1 + i]; }
}

int main(int argc, char **args)
{
  Mat            A;
  Vec            b, x, r, p;
  Vec           *B; // Basis vectors [P0..Ps, R0..Rs-1]
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscLogDouble t1, t2;
  PetscViewer    viewer;
  char           file[PETSC_MAX_PATH_LEN];
  PetscScalar    tol      = 1e-6;
  PetscInt       s        = 5;
  PetscInt       max_iter = 10000;
  PetscInt       iter, k, j, i, dim;
  PetscMPIInt    rank;

  // Scalar loop variables
  PetscScalar *G;            // Gram matrix V_l^T*V_l; (2s+1)*(2s+1)
  PetscScalar *pc, *rc, *xc; // Coordinate vectors - direction, residual, solution
  PetscScalar *pc_next, *rc_next, *temp_vec;
  PetscScalar  alpha, beta;
  PetscReal    r_norm, r0_norm;

  ierr = PetscInitialize(&argc, &args, (char *)0, help);
  CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Option flags
  ierr = PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg);
  CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate matrix file with -f <filename>");
  ierr = PetscOptionsGetScalar(NULL, NULL, "-tol", &tol, &flg);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-s", &s, &flg);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, &flg);
  CHKERRQ(ierr);

  // Load matrix
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer);
  CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);
  CHKERRQ(ierr);
  ierr = MatLoad(A, viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);
  CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_SPD, PETSC_TRUE);

  // Vectors
  ierr = MatCreateVecs(A, &x, &b);
  CHKERRQ(ierr);
  ierr = VecSet(b, 1.0);
  CHKERRQ(ierr);
  ierr = VecSet(x, 0.0);
  CHKERRQ(ierr);

  ierr = VecDuplicate(b, &r);
  CHKERRQ(ierr);
  ierr = VecDuplicate(b, &p);
  CHKERRQ(ierr);

  // Init variables
  // r_1 := b - Ax_1; (x_1=0, r=b)
  ierr = VecCopy(b, r);
  CHKERRQ(ierr);
  // p_1 := r_1
  ierr = VecCopy(r, p);
  CHKERRQ(ierr);

  // Allocate basis vectors B of size 2s+1, store the combined basis V_l
  dim  = 2 * s + 1;
  ierr = VecDuplicateVecs(b, dim, &B);
  CHKERRQ(ierr);

  // Allocate arrays
  ierr = PetscMalloc1(dim * dim, &G);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &pc);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &rc);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &xc);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &pc_next);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &rc_next);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &temp_vec);
  CHKERRQ(ierr);

  // Timesttamp @ t_1
  ierr = PetscTime(&t1);
  CHKERRQ(ierr);

  // Error
  ierr = VecNorm(r, NORM_2, &r0_norm);
  CHKERRQ(ierr);
  r_norm = r0_norm;
  if (r0_norm == 0.0) r0_norm = 1.0;

  if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Starting CACG (s=%d) | Initial Norm: %g\n", s, (double)r_norm);

  iter               = 0;
  PetscInt breakdown = 0;

  // Outer loop: for j += s do
  while (iter < max_iter && r_norm > tol) {
    // Generate basis
    // P_j := p_j (stored in B[0])
    ierr = VecCopy(p, B[0]);
    CHKERRQ(ierr);

    // P_{j+1} := A * P_j (Loop computes P basis)
    for (i = 0; i < s; i++) {
      ierr = MatMult(A, B[i], B[i + 1]);
      CHKERRQ(ierr);
    }

    // R_j := r_j (Stored in B[s+1])
    ierr = VecCopy(r, B[s + 1]);
    CHKERRQ(ierr);

    // R_{j+k} := A * R_{j+k-1}
    for (i = 0; i < s - 1; i++) {
      ierr = MatMult(A, B[s + 1 + i], B[s + 1 + i + 1]);
      CHKERRQ(ierr);
    }

    // Gram matrix computatio: G := V_l^T * V_l
    for (i = 0; i < dim; i++) {
      ierr = VecMDot(B[i], dim, B, &G[i * dim]);
      CHKERRQ(ierr);
    }

    // Not in paper: Broadcast G to ensure bitwise identity across ranks. Necessary for stability.
    MPI_Bcast(G, dim * dim, MPIU_SCALAR, 0, PETSC_COMM_WORLD);

    // Initialize coordinate vectors
    // c_1 := e_1
    for (i = 0; i < dim; i++) {
      pc[i] = 0.0;
      rc[i] = 0.0;
      xc[i] = 0.0;
    }
    pc[0] = 1.0;
    // t_1 := e_{s+1}
    rc[s + 1] = 1.0;

    // Inner loop: for k = 1 to s
    for (j = 0; j < s && iter < max_iter; j++) {
      iter++;

      // Compute alpha
      // gamma_k := c_k^T * G * d_k (where d_k = A*c_k, handled by T shift)
      // delta_k := t_k^T * G * t_k

      // Compute numerator: r_c^T * G * r_c (delta_k)
      PetscScalar delta = 0.0;
      for (int row = 0; row < dim; row++) {
        PetscScalar row_val = 0.0;
        for (int col = 0; col < dim; col++) row_val += G[row * dim + col] * rc[col];
        delta += rc[row] * row_val;
      }

      // Compute denominator: p_c^T * G * T * p_c (gamma_k)
      ApplyMonomialT(s, pc, temp_vec); // temp_vec = T * p_c (d_k)

      PetscScalar gamma = 0.0;
      for (int row = 0; row < dim; row++) {
        PetscScalar row_val = 0.0;
        for (int col = 0; col < dim; col++) row_val += G[row * dim + col] * temp_vec[col];
        gamma += pc[row] * row_val;
      }

      // Check for breakdown
      if (PetscAbsScalar(gamma) < 1e-16) {
        breakdown = 1;
        break;
      }

      // alpha_k := delta_k / gamma_k
      alpha = delta / gamma;

      // Update vectors
      // y_{k+1} := y_k + alpha_k * c_k
      for (i = 0; i < dim; i++) xc[i] += alpha * pc[i];

      // t_{k+1} := t_k - alpha_k * d_k
      for (i = 0; i < dim; i++) rc_next[i] = rc[i] - alpha * temp_vec[i];

      // Compute Beta
      // delta_{k+1} := t_{k+1}^T * G * t_{k+1}
      PetscScalar beta_num = 0.0;
      for (int row = 0; row < dim; row++) {
        PetscScalar row_val = 0.0;
        for (int col = 0; col < dim; col++) row_val += G[row * dim + col] * rc_next[col];
        beta_num += rc_next[row] * row_val;
      }

      // beta_k := delta_{k+1} / delta_k
      if (PetscAbsScalar(delta) < 1e-16) beta = 0.0; // Handle low resid
      else beta = beta_num / delta;

      // Update direction
      // c_{k+1} := t_{k+1} + beta_k * c_k
      for (i = 0; i < dim; i++) pc_next[i] = rc_next[i] + beta * pc[i];

      // Copy next to curr for next step
      for (i = 0; i < dim; i++) {
        rc[i] = rc_next[i];
        pc[i] = pc_next[i];
      }

      // Error
      r_norm = PetscSqrtReal(PetscRealPart(beta_num));
    }

    if (breakdown) {
      if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Breakdown: Gram matrix singularity. Monomial basis unstable.\n");
      break;
    }

    // Reconstruction
    // Sync coordinate vectors to ensure parallel consistency
    MPI_Bcast(xc, dim, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    MPI_Bcast(rc, dim, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    MPI_Bcast(pc, dim, MPIU_SCALAR, 0, PETSC_COMM_WORLD);

    // x_{j+s} := x_j + [P, R] * y_{s+1}
    ierr = VecMAXPY(x, dim, xc, B);
    CHKERRQ(ierr);

    // r_{j+s} := [P, R] * t_{s+1}
    ierr = VecSet(r, 0.0);
    CHKERRQ(ierr);
    ierr = VecMAXPY(r, dim, rc, B);
    CHKERRQ(ierr);

    // p_{j+s} := [P, R] * c_{s+1}
    ierr = VecSet(p, 0.0);
    CHKERRQ(ierr);
    ierr = VecMAXPY(p, dim, pc, B);
    CHKERRQ(ierr);

    // Print progress
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Iter %d, Resid %g\n", iter, (double)r_norm);

    // Convergence check
    if (r_norm / r0_norm <= tol || PetscIsInfOrNanScalar(r_norm)) break;
  }

  // Timesttamp @ t_2
  ierr = PetscTime(&t2);
  CHKERRQ(ierr);

  // Results
  if (rank == 0) {
    if (breakdown) {
      PetscPrintf(PETSC_COMM_WORLD, "FAILURE: Algorithm breakdown.\n");
    } else if (r_norm / r0_norm <= tol) {
      PetscPrintf(PETSC_COMM_WORLD, "Converged in %d iterations, Time: %g\n", iter, t2 - t1);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "Failed to converge. Resid: %g\n", (double)r_norm);
    }
  }

  // Cleanup
  ierr = VecDestroyVecs(dim, &B);
  CHKERRQ(ierr);
  ierr = VecDestroy(&x);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = VecDestroy(&r);
  CHKERRQ(ierr);
  ierr = VecDestroy(&p);
  CHKERRQ(ierr);
  ierr = MatDestroy(&A);
  CHKERRQ(ierr);

  ierr = PetscFree(G);
  ierr = PetscFree(pc);
  ierr = PetscFree(rc);
  ierr = PetscFree(xc);
  ierr = PetscFree(pc_next);
  ierr = PetscFree(rc_next);
  ierr = PetscFree(temp_vec);

  ierr = PetscFinalize();
  return 0;
}
