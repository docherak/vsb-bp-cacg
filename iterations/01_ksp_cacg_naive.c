#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petsc.h>
#include <petscksp.h>
#include <petsc/private/kspimpl.h>

// 1. Context Structure
typedef struct {
  PetscInt  s; // The s-step parameter
  PetscBool alloc_done;

  // DISTRIBUTED
  // Basis V = [P, R]
  // s + 1 vectors of P
  // s vectirs of R
  // G = V^T * V
  Mat V;

  // Vector views into V
  // Pointers to the columns of V
  // KSP_MatMult() requires Vec not Mat
  // Vec *P;
  // Vec *R;

  // GRAM MATRICES
  // 1. Distributed
  // Result of MatTransposeMatMult(V, V)
  Mat G_dist;
  // 2. Replicated
  // Result of MatCreateRedundantMatrix(G_dist)
  // Copy on every rank
  Mat G_repl;

  // COORDINATE VECTORS
  // Locally / redundantly on every rank
  Vec x, r, p;
  // Helper for Monomial Shift
  Vec mon_tmp;
  // Helper for SpMV safety (Distributed scratchpad)
  // We need this because we cannot lock two columns of V simultaneously (prev + next)
  Vec v_tmp;
} KSP_CACG;

// 2. Helper functions
// Simulates w = A * v in the coordinate basis.
// For Monomial basis: Shifts P coefficients (0->1) and R coefficients (s+1->s+2)
static PetscErrorCode ApplyMonomialShift(PetscInt s, Vec v_in, Vec v_out)
{
  PetscScalar *in, *out;

  PetscFunctionBegin;
  PetscCall(VecSet(v_out, 0.0)); // Clear output
  PetscCall(VecGetArray(v_in, &in));
  PetscCall(VecGetArray(v_out, &out));

  // Shift P: indices 0..(s-1) -> 1..s
  for (int i = 0; i < s; i++) out[i + 1] += in[i];

  // Shift R: indices (s+1)..(2s-1) -> (s+2)..2s
  for (int i = 0; i < s - 1; i++) out[s + 1 + i + 1] += in[s + 1 + i];

  PetscCall(VecRestoreArray(v_in, &in));
  PetscCall(VecRestoreArray(v_out, &out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 3. Operations

static PetscErrorCode KSPSetUp_CACG(KSP ksp)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;
  if (cg->s < 1) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "s-step size must be >= 1");

  // Future Logic: If s=1, we could just setup standard CG structures
  // For now, we will implement the full s-step logic even for s=1 to prove correctness.
  PetscInt s   = cg->s;
  PetscInt dim = 2 * s + 1; // Total basis size: (s+1) for P, s for R
  PetscInt m_local, M_global;
  Mat      A; // System matrix

  PetscFunctionBegin;

  // Get dimensions from A
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatGetLocalSize(A, &m_local, NULL));
  PetscCall(MatGetSize(A, &M_global, NULL));

  // Alloc Distributed basis V
  // MPIDense
  if (!cg->V) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp), // Communicator
                             m_local,                           // Local Rows
                             PETSC_DECIDE,                      // Local Cols
                             M_global,                          // Global Rows
                             dim,                               // Global Cols
                             NULL,                              // Data Pointer
                             &cg->V                             // Output
                             ));
  }

  // Alloc Replicated vectors (2s + 1), on CPU
  if (!cg->p) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->p));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->r));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->mon_tmp));
  }

  // Allocate ctTemporary helper vector
  // Must match layout of the system matrix A (Distributed)
  if (!cg->v_tmp) { PetscCall(MatCreateVecs(A, &cg->v_tmp, NULL)); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_CACG(KSP ksp)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;

  // Future Logic:
  // if (cg->s == 1) {
  //     return KSPSolve_CG(ksp); // Fallback to optimized PETSc CG
  // }

  Mat       A;
  Vec       x, b, r, p;
  PetscInt  s = cg->s;
  PetscReal r_norm;
  PetscInt  i;

  PetscFunctionBegin;

  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &b));

  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecDuplicate(b, &p));

  // Initial residual r = b - A*x
  PetscCall(KSP_MatMult(ksp, A, x, r));
  PetscCall(VecAYPX(r, -1.0, b)); // r = b - r, i.e b - Ax
  PetscCall(VecCopy(r, p));       // p = r

  // Initial convergence
  PetscCall(VecNorm(r, NORM_2, &r_norm));
  ksp->rnorm = r_norm;
  ksp->its   = 0;

  // Log initial status
  PetscCall(KSPMonitor(ksp, 0, r_norm));
  PetscCall((*ksp->converged)(ksp, 0, r_norm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  // Outer loop: every 's'-th iteration
  while (ksp->its < ksp->max_it && !ksp->reason) {
    // GENERATE BASIS V = [P, R]
    //
    // A. P-Basis (0 to s)
    {
      Vec col_vec; // to save return from MatDenseGetColumnVecWrite
      // 1. Write p into V[0]
      PetscCall(MatDenseGetColumnVecWrite(cg->V, 0, &col_vec));
      PetscCall(VecCopy(p, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(cg->V, 0, &col_vec));
      // 2. Krylov Chain: P_{i+1} = A * P_i
      for (i = 0; i < s; i++) {
        Vec p_prev, p_next;

        // // Get view of col i (input)
        // PetscCall(MatDenseGetColumnVecRead(cg->V, i, &p_prev));
        // // Get view of col i+1 (output)
        // PetscCall(MatDenseGetColumnVecWrite(cg->V, i + 1, &p_next));

        // // p_next = A * p_prev
        // PetscCall(KSP_MatMult(ksp, A, p_prev, p_next));

        // // Restore views
        // PetscCall(MatDenseRestoreColumnVecRead(cg->V, i, &p_prev));
        // PetscCall(MatDenseRestoreColumnVecWrite(cg->V, i + 1, &p_next));

        // Step 1: Compute next vector into v_tmp
        // Lock Column i
        PetscCall(MatDenseGetColumnVecRead(cg->V, i, &p_prev));
        // v_tmp = A * p_prev
        PetscCall(KSP_MatMult(ksp, A, p_prev, cg->v_tmp));
        // Unlock Column i (CRITICAL: Must unlock before getting next column)
        PetscCall(MatDenseRestoreColumnVecRead(cg->V, i, &p_prev));

        // Step 2: Copy v_tmp into Column i+1
        // Lock Column i+1
        PetscCall(MatDenseGetColumnVecWrite(cg->V, i + 1, &p_next));
        // p_next = v_tmp
        PetscCall(VecCopy(cg->v_tmp, p_next));
        // Unlock Column i+1
        PetscCall(MatDenseRestoreColumnVecWrite(cg->V, i + 1, &p_next));
      }
    }
    // A. R-Basis (s+1 to 2s)
    {
      Vec col_vec; // to save return from MatDenseGetColumnVecWrite
      // 1. Write r into V[s+1]
      PetscCall(MatDenseGetColumnVecWrite(cg->V, s + 1, &col_vec));
      PetscCall(VecCopy(r, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(cg->V, s + 1, &col_vec));
      // 2. Krylov Chain: R_{i+1} = A * R_i
      for (i = 0; i < s - 1; i++) {
        Vec r_prev, r_next;

        // // Get view of col i (input)
        // PetscCall(MatDenseGetColumnVecRead(cg->V, s + 1 + i, &r_prev));
        // // Get view of col i+1 (output)
        // PetscCall(MatDenseGetColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));

        // // r_next = A * r_prev
        // PetscCall(KSP_MatMult(ksp, A, r_prev, r_next));

        // // Restore views
        // PetscCall(MatDenseRestoreColumnVecRead(cg->V, s + 1 + i, &r_prev));
        // PetscCall(MatDenseRestoreColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));

        // Step 1: Compute into v_tmp
        PetscCall(MatDenseGetColumnVecRead(cg->V, s + 1 + i, &r_prev));
        PetscCall(KSP_MatMult(ksp, A, r_prev, cg->v_tmp));
        PetscCall(MatDenseRestoreColumnVecRead(cg->V, s + 1 + i, &r_prev));

        // Step 2: Copy to next column
        PetscCall(MatDenseGetColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));
        PetscCall(VecCopy(cg->v_tmp, r_next));
        PetscCall(MatDenseRestoreColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));
      }
    }

    // GRAM MATRIX COMPUTATION: G = V^T * V
    // Reuse flag - 0th iter - create matrix, iter > 0 -> reuse
    MatReuse reuse = (cg->G_dist) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

    // Compute Distributed Gram
    // input cg->V
    // output cg->G_dist
    PetscCall(MatTransposeMatMult(cg->V, cg->V, reuse, PETSC_DEFAULT, &cg->G_dist));

    // Global Reduction
    // input cg->G_dist
    // output cg->G_repl
    PetscCall(MatCreateRedundantMatrix(cg->G_dist, 0, PETSC_COMM_SELF, reuse, &cg->G_repl));

    // INNER LOOP - CPU math
    // Reset coords
    // x_c = 0
    // p_c = e_1
    // r_c = e_{s+1}

    // 1. Reset Coordinate Vectors
    // x_c starts at 0 (accumulates updates)
    PetscCall(VecSet(cg->x, 0.0));

    // p_c = e_1 (The first vector in P-block is p)
    PetscCall(VecSet(cg->p, 0.0));
    PetscCall(VecSetValue(cg->p, 0, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(cg->p));
    PetscCall(VecAssemblyEnd(cg->p));

    // r_c = e_{s+1} (The first vector in R-block is r)
    PetscCall(VecSet(cg->r, 0.0));
    PetscCall(VecSetValue(cg->r, s + 1, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(cg->r));
    PetscCall(VecAssemblyEnd(cg->r));

    // 2. The s-step Inner Loop
    // Runs 's' times entirely in local memory (Replicated)
    for (int j = 0; j < s && ksp->its < ksp->max_it; j++) {
      ksp->its++;
      PetscScalar delta, gamma, alpha, beta;

      // --- A. Compute Delta = r_c^T * G * r_c ---
      // We use mon_tmp as scratch for (G * r_c)
      PetscCall(MatMult(cg->G_repl, cg->r, cg->mon_tmp)); // mon_tmp = G * r_c
      PetscCall(VecDot(cg->r, cg->mon_tmp, &delta));      // delta = r_c . mon_tmp

      // --- B. Compute Gamma = p_c^T * G * (T * p_c) ---
      // 1. Apply Shift T: mon_tmp = T * p_c
      PetscCall(ApplyMonomialShift(s, cg->p, cg->mon_tmp));

      // 2. Create tiny scratch vector t_vec on the fly
      Vec t_vec;
      PetscCall(VecDuplicate(cg->p, &t_vec));

      // 3. t_vec = G * (T * p_c)
      PetscCall(MatMult(cg->G_repl, cg->mon_tmp, t_vec));

      // 4. gamma = p_c . t_vec
      PetscCall(VecDot(cg->p, t_vec, &gamma));

      // Check Breakdown (Division by zero)
      if (PetscAbsScalar(gamma) == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscCall(VecDestroy(&t_vec));
        break;
      }

      // --- C. Calculate Alpha ---
      alpha = delta / gamma;

      // --- D. Update Coordinates ---
      // x_c = x_c + alpha * p_c
      PetscCall(VecAXPY(cg->x, alpha, cg->p));

      // r_c = r_c - alpha * (T * p_c)
      // Note: mon_tmp currently holds (T * p_c) from Step B.1
      PetscCall(ApplyMonomialShift(s, cg->p, cg->mon_tmp)); // Re-compute to be safe
      PetscCall(VecAXPY(cg->r, -alpha, cg->mon_tmp));

      // --- E. Calculate Beta ---
      // delta_new = r_c_new^T * G * r_c_new
      PetscCall(MatMult(cg->G_repl, cg->r, t_vec)); // t_vec = G * r_c
      PetscScalar delta_new;
      PetscCall(VecDot(cg->r, t_vec, &delta_new));

      beta = delta_new / delta;

      // --- F. Update p_c ---
      // p_c = r_c + beta * p_c
      PetscCall(VecAYPX(cg->p, beta, cg->r));

      // Monitor convergence (approximate)
      r_norm = PetscSqrtReal(PetscRealPart(delta_new));
      PetscCall(KSPMonitor(ksp, ksp->its, r_norm));

      // Cleanup scratch
      PetscCall(VecDestroy(&t_vec));

      // Check tolerance
      if (r_norm < ksp->rtol * 1.0) { break; }
    }
    //   // RECONSTRUCTION: Coordinate -> Global

    //   // 1. Update Global Solution: x = x + V * x_c
    //   // cg->x is the coordinate vector (size 2s+1).
    //   // cg->V is the distributed basis.
    //   PetscCall(MatMultAdd(cg->V, cg->x, x, x));

    //   // 2. Recompute Exact Residual (Standard Stabilization)
    //   // r = b - A * x
    //   PetscCall(KSP_MatMult(ksp, A, x, r));
    //   PetscCall(VecAYPX(r, -1.0, b));

    //   // 3. Reconstruct Direction p
    //   // p = V * p_c
    //   // Note: cg->p holds the coordinates of the NEW direction at the end of the inner loop.
    //   PetscCall(MatMult(cg->V, cg->p, p));

    //   // 4. Final Norm Check
    //   PetscCall(VecNorm(r, NORM_2, &r_norm));
    //   ksp->rnorm = r_norm;

    //   // Check Outer Loop Convergence
    //   PetscCall((*ksp->converged)(ksp, ksp->its, r_norm, &ksp->reason, ksp->cnvP));
    // }

    // PetscCall(VecDestroy(&r));
    // PetscCall(VecDestroy(&p));

    // PetscFunctionReturn(PETSC_SUCCESS);
    //
    // RECONSTRUCTION

    Mat          V_local;
    Vec          x_local_wrapper, p_local_wrapper;
    PetscScalar *x_ptr, *p_ptr;
    PetscInt     m_local_rows;

    // Get Local Matrix from V (Borrowed reference, do not destroy)
    PetscCall(MatDenseGetLocalMatrix(cg->V, &V_local));
    PetscCall(MatGetLocalSize(cg->V, &m_local_rows, NULL));

    // A. Reconstruct x = x + V * x_c
    // 1. Get pointer to local data of MPI vector x
    PetscCall(VecGetArray(x, &x_ptr));
    // 2. Wrap it in a Sequential Vector
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, x_ptr, &x_local_wrapper));
    // 3. Do local multiplication: x_local = x_local + V_local * cg->x
    PetscCall(MatMultAdd(V_local, cg->x, x_local_wrapper, x_local_wrapper));
    // 4. Cleanup wrapper and restore array
    PetscCall(VecDestroy(&x_local_wrapper));
    PetscCall(VecRestoreArray(x, &x_ptr));

    // B. Reconstruct p = V * p_c
    // cg->p now holds coordinates for the NEXT direction
    PetscCall(VecGetArray(p, &p_ptr));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, p_ptr, &p_local_wrapper));
    PetscCall(MatMult(V_local, cg->p, p_local_wrapper));
    PetscCall(VecDestroy(&p_local_wrapper));
    PetscCall(VecRestoreArray(p, &p_ptr));

    // // Recompute True Residual r = b - Ax
    // PetscCall(KSP_MatMult(ksp, A, x, r));
    // PetscCall(VecAYPX(r, -1.0, b));
    //

    {
      PetscScalar *r_ptr;
      Vec          r_local_wrapper;
      PetscInt     m_local_rows;

      // Get local size of the dense basis V to match dimensions
      PetscCall(MatGetLocalSize(cg->V, &m_local_rows, NULL));

      // Get raw access to the global residual vector 'r'
      PetscCall(VecGetArray(r, &r_ptr));

      // Create a temporary wrapper to treat the array 'r_ptr' as a local sequential vector
      // This allows us to use MatMult with the local dense part of V
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, r_ptr, &r_local_wrapper));

      // Compute: r_local = V_local * t_final
      // Note: cg->r holds the coordinate vector 't' from the inner loop
      PetscCall(MatMult(V_local, cg->r, r_local_wrapper));

      // Cleanup
      PetscCall(VecDestroy(&r_local_wrapper));
      PetscCall(VecRestoreArray(r, &r_ptr));
    }

    PetscCall(VecNorm(r, NORM_2, &r_norm));
    ksp->rnorm = r_norm;
    PetscCall((*ksp->converged)(ksp, ksp->its, r_norm, &ksp->reason, ksp->cnvP));
  }
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS; 

  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_CACG(KSP ksp)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;

  PetscFunctionBegin;

  // 1. Destroy Replicated Coordinate Vectors
  PetscCall(VecDestroy(&cg->p));
  PetscCall(VecDestroy(&cg->r));
  PetscCall(VecDestroy(&cg->x));
  PetscCall(VecDestroy(&cg->mon_tmp));

  // 2. Destroy Matrices
  PetscCall(MatDestroy(&cg->V));
  PetscCall(MatDestroy(&cg->G_dist));
  PetscCall(MatDestroy(&cg->G_repl));

  // 3. Free the Context Structure
  // Removes the memory allocated by PetscNew(&cg)
  PetscCall(PetscFree(ksp->data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_CACG(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP CACG Options");
  PetscCall(PetscOptionsInt("-ksp_cacg_s", "Number of s-steps (s)", "KSPCACG", cg->s, &cg->s, NULL));
  PetscOptionsHeadEnd();
  return 0;
}

// 3. Factory Function (Registration)
PETSC_EXTERN PetscErrorCode KSPCreate_CACG(KSP ksp)
{
  KSP_CACG *cg;
  PetscNew(&cg);
  ksp->data = (void *)cg;

  // Default s=1 (Behaves like standard CG mathematically)
  cg->s = 1;

  // Binding Ops
  ksp->ops->setup          = KSPSetUp_CACG;
  ksp->ops->solve          = KSPSolve_CACG;
  ksp->ops->destroy        = KSPDestroy_CACG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CACG;

  // Standard CG norm support
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));

  return 0;
}

// 4. Main Driver
int main(int argc, char **args)
{
  Vec         x, b, u;
  Mat         A;
  KSP         ksp;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   flg;
  PetscInt    i, n = 100, col[3];
  PetscScalar value[3];
  PetscReal   norm;
  PetscViewer viewer;

  PetscLogStage stage_cacg;

  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscCall(KSPRegister("cacg", KSPCreate_CACG));

  PetscCall(PetscLogStageRegister("Solve_CACG", &stage_cacg));

  // ------------------------------------------------------------
  // 1. LOAD / GENERATE MATRIX
  // ------------------------------------------------------------
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));

  if (flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Loading matrix from %s...\n", file));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecDuplicate(x, &u));

  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Generating 1D Laplacian...\n"));

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));

    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecDuplicate(x, &u));

    PetscInt rstart, rend;
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

    for (i = rstart; i < rend; i++) {
      if (i == 0) {
        col[0]   = 0;
        col[1]   = 1;
        value[0] = 2.0;
        value[1] = -1.0;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
      } else if (i == n - 1) {
        col[0]   = n - 2;
        col[1]   = n - 1;
        value[0] = -1.0;
        value[1] = 2.0;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
      } else {
        col[0]   = i - 1;
        col[1]   = i;
        col[2]   = i + 1;
        value[0] = -1.0;
        value[1] = 2.0;
        value[2] = -1.0;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
      }
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  // ------------------------------------------------------------
  // 2. PROBLEM PREPARATION
  // ------------------------------------------------------------
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  // ------------------------------------------------------------
  // 3. SOLVE
  // ------------------------------------------------------------
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> STARTING CACG <<<\n"));

  PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, "cacg"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PetscLogStagePush(stage_cacg));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(PetscLogStagePop());

  // ------------------------------------------------------------
  // 4. ERROR CHECK
  // ------------------------------------------------------------
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &i));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CACG RESULT: Iters %" PetscInt_FMT ", Err %g\n", i, (double)norm));

  // Cleanup
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
