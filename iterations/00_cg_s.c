#include <petsc.h>
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
  Mat         A, B_mat, G_mat = NULL, G_seq = NULL;
  Vec         b, x, r, p, x_true, e_vec, temp_vec_A;
  Vec         y_vec; // Helper vector for reconstruction
  Vec         v_tmp; // Helper vector for safe column copy
  PetscBool   flg;
  PetscViewer viewer;
  char        file[PETSC_MAX_PATH_LEN];
  PetscScalar tol      = 1e-6;
  PetscInt    s        = 5;
  PetscInt    max_iter = 10000;
  PetscInt    iter, k, j, i, dim;
  PetscMPIInt rank;

  // Scalar loop variables
  PetscScalar  *G;            // Gram matrix buffer
  PetscScalar  *pc, *rc, *xc; // Coordinate vectors - direction, residual, solution
  PetscScalar  *pc_next, *rc_next, *temp_vec;
  PetscScalar   alpha, beta;
  PetscReal     r_norm, b_norm;
  PetscReal     err_A_norm;
  PetscScalar   err_dot;
  PetscLogEvent event_solve;
  PetscInt     *indices; // For VecSetValues

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  // Option flags
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate matrix file with -f <filename>");
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-tol", &tol, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, &flg));
  dim = 2 * s + 1;

  // Load matrix
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));

  // RHS from known x
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &x_true));
  PetscCall(VecSet(x_true, 1.0));        // Set entries to 1
  PetscCall(VecNormalize(x_true, NULL)); // Normalize vector
  PetscCall(MatMult(A, x_true, b));      // b = A * x_true

  // Initialize solver vectors
  PetscCall(VecSet(x, 0.0)); // Initial guess x0 = 0
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecDuplicate(b, &p));

  // Init variables
  // r_1 := b - Ax_1; (x_1=0, r=b)
  PetscCall(VecCopy(b, r));
  // p_1 := r_1
  PetscCall(VecCopy(r, p));

  // Temp vector to avoid double lock on B_mat
  PetscCall(VecDuplicate(b, &v_tmp));

  // Allocate basis B as a dense matrix
  PetscInt m_local, M_global;
  PetscCall(MatGetLocalSize(A, &m_local, NULL));
  PetscCall(MatGetSize(A, &M_global, NULL));

  // B_mat: m_local rows, dim columns
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m_local, PETSC_DECIDE, M_global, dim, NULL, &B_mat));

  // Helper vector 'y_vec' compatible with the columns of B, used to hold coordinate vectors (xc, rc, pc) for reconstruction
  PetscCall(MatCreateVecs(B_mat, &y_vec, NULL));

  // Allocate arrays
  PetscCall(PetscMalloc1(dim * dim, &G));
  PetscCall(PetscMalloc1(dim, &pc));
  PetscCall(PetscMalloc1(dim, &rc));
  PetscCall(PetscMalloc1(dim, &xc));
  PetscCall(PetscMalloc1(dim, &pc_next));
  PetscCall(PetscMalloc1(dim, &rc_next));
  PetscCall(PetscMalloc1(dim, &temp_vec));

  // Indices 0..dim-1 for VecSetValues
  PetscCall(PetscMalloc1(dim, &indices));
  for (i = 0; i < dim; i++) indices[i] = i;

  // Calculate b_norm for relative tolerance
  PetscCall(VecNorm(b, NORM_2, &b_norm));
  if (b_norm == 0.0) b_norm = 1.0;

  // Calculate initial residual norm
  PetscCall(VecNorm(r, NORM_2, &r_norm));

  if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Starting CACG (s=%d) | Initial Norm: %g | Target Rel Tol: %g\n", s, (double)r_norm, (double)tol);

  // Profiling Setup
  PetscCall(PetscLogEventRegister("s-step Solve", 0, &event_solve));
  PetscCall(PetscLogEventBegin(event_solve, 0, 0, 0, 0));

  iter               = 0;
  PetscInt breakdown = 0;

  // Outer loop: for j += s do
  // Relative tolerance: r_norm > tol * b_norm
  while (iter < max_iter && r_norm > tol * b_norm) {
    // -----
    // 1. Generate Basis
    // -----

    // P_j := p_j (stored in col 0)
    {
      Vec col_vec;
      PetscCall(MatDenseGetColumnVecWrite(B_mat, 0, &col_vec));
      PetscCall(VecCopy(p, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(B_mat, 0, &col_vec));
    }

    // P_{j+1} := A * P_j (Loop computes P basis)
    for (i = 0; i < s; i++) {
      Vec p_prev, p_next;

      PetscCall(MatDenseGetColumnVecRead(B_mat, i, &p_prev));
      PetscCall(VecCopy(p_prev, v_tmp));
      PetscCall(MatDenseRestoreColumnVecRead(B_mat, i, &p_prev));

      PetscCall(MatDenseGetColumnVecWrite(B_mat, i + 1, &p_next));
      PetscCall(MatMult(A, v_tmp, p_next));
      PetscCall(MatDenseRestoreColumnVecWrite(B_mat, i + 1, &p_next));
    }

    // R_j := r_j (Stored in col s+1)
    {
      Vec col_vec;
      PetscCall(MatDenseGetColumnVecWrite(B_mat, s + 1, &col_vec));
      PetscCall(VecCopy(r, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(B_mat, s + 1, &col_vec));
    }

    // R_{j+k} := A * R_{j+k-1}
    for (i = 0; i < s - 1; i++) {
      Vec r_prev, r_next;

      PetscCall(MatDenseGetColumnVecRead(B_mat, s + 1 + i, &r_prev));
      PetscCall(VecCopy(r_prev, v_tmp));
      PetscCall(MatDenseRestoreColumnVecRead(B_mat, s + 1 + i, &r_prev));

      PetscCall(MatDenseGetColumnVecWrite(B_mat, s + 1 + i + 1, &r_next));
      PetscCall(MatMult(A, v_tmp, r_next));
      PetscCall(MatDenseRestoreColumnVecWrite(B_mat, s + 1 + i + 1, &r_next));
    }

    // -----
    // 2. Gram Matrix Computation (Optimized)
    // G = B^T * B
    // -----

    MatReuse reuse = (G_mat) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

    PetscCall(MatTransposeMatMult(B_mat, B_mat, reuse, PETSC_DEFAULT, &G_mat));

    // Gather distributed G_mat to local sequential G_seq on all ranks
    PetscCall(MatCreateRedundantMatrix(G_mat, 0, PETSC_COMM_SELF, reuse, &G_seq));

    // Copy raw memory directly to G array
    const PetscScalar *g_ptr;
    PetscCall(MatDenseGetArrayRead(G_seq, &g_ptr));
    PetscCall(PetscMemcpy(G, g_ptr, dim * dim * sizeof(PetscScalar)));
    PetscCall(MatDenseRestoreArrayRead(G_seq, &g_ptr));

    // -----
    // 3. Scalar Inner Loop
    // -----

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

    // -----
    // 4. Reconstruction
    // -----
    // MatMult then performs the reconstruction: x = x + B * xc

    // x_{j+s} := x_j + [P, R] * y_{s+1}
    PetscCall(VecSetValues(y_vec, dim, indices, xc, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(y_vec));
    PetscCall(VecAssemblyEnd(y_vec));
    PetscCall(MatMultAdd(B_mat, y_vec, x, x));

    // r_{j+s} := [P, R] * t_{s+1}
    PetscCall(VecSetValues(y_vec, dim, indices, rc, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(y_vec));
    PetscCall(VecAssemblyEnd(y_vec));
    PetscCall(MatMult(B_mat, y_vec, r));

    // p_{j+s} := [P, R] * c_{s+1}
    PetscCall(VecSetValues(y_vec, dim, indices, pc, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(y_vec));
    PetscCall(VecAssemblyEnd(y_vec));
    PetscCall(MatMult(B_mat, y_vec, p));

    // Print progress
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Iter %d, Resid %g\n", iter, (double)r_norm);

    // Convergence check
    if (r_norm < tol * b_norm || PetscIsInfOrNanScalar(r_norm)) break;
  }

  PetscCall(PetscLogEventEnd(event_solve, 0, 0, 0, 0));

  // Compute A-norm of error
  // ||x_true - x||_A = sqrt( (x_true-x)^T * A * (x_true-x) )
  PetscCall(VecDuplicate(x, &e_vec));
  PetscCall(VecDuplicate(x, &temp_vec_A));

  // e = x_true - x
  PetscCall(VecCopy(x_true, e_vec));
  PetscCall(VecAXPY(e_vec, -1.0, x));

  // temp = A * e
  PetscCall(MatMult(A, e_vec, temp_vec_A));

  // dot = e . temp
  PetscCall(VecDot(e_vec, temp_vec_A, &err_dot));
  err_A_norm = PetscSqrtReal(PetscRealPart(err_dot));

  // Results
  if (rank == 0) {
    if (breakdown) {
      PetscPrintf(PETSC_COMM_WORLD, "FAILURE: Algorithm breakdown.\n");
    } else if (r_norm < tol * b_norm) {
      PetscPrintf(PETSC_COMM_WORLD, "Converged in %d iterations.\n", iter);
      PetscPrintf(PETSC_COMM_WORLD, "A-norm of Error: %g\n", (double)err_A_norm);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "Failed to converge. Resid: %g\n", (double)r_norm);
    }
  }

  // Cleanup
  PetscCall(MatDestroy(&B_mat));
  PetscCall(MatDestroy(&G_mat));
  PetscCall(MatDestroy(&G_seq));
  PetscCall(VecDestroy(&y_vec));
  PetscCall(VecDestroy(&v_tmp));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&x_true));
  PetscCall(VecDestroy(&e_vec));
  PetscCall(VecDestroy(&temp_vec_A));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFree(G));
  PetscCall(PetscFree(pc));
  PetscCall(PetscFree(rc));
  PetscCall(PetscFree(xc));
  PetscCall(PetscFree(pc_next));
  PetscCall(PetscFree(rc_next));
  PetscCall(PetscFree(temp_vec));
  PetscCall(PetscFree(indices));

  PetscCall(PetscFinalize());
  return 0;
}
