#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petsc.h>
#include <petscksp.h>
#include <petsc/private/kspimpl.h>
#include <petscblaslapack.h>

#define KSPCACG "cacg"

// --------------------------------------------------------
// GLOBAL LOG EVENTS
// --------------------------------------------------------
static PetscLogEvent KSP_CACG_BasisGen;
static PetscLogEvent KSP_CACG_Gram;
static PetscLogEvent KSP_CACG_InnerLoop;
static PetscLogEvent KSP_CACG_Reconstruct;

typedef struct {
  PetscInt s;
  Mat      V;
  Mat      G_repl;
  Vec      x, r, p;
  Vec      mon_tmp, t_vec;
  Vec      v_tmp;
} KSP_CACG;

// Simple matrix-vector multiply for tiny dimension (dim x dim) * (dim x 1)
static inline void TinyMatMult(PetscInt dim, const PetscScalar *G, const PetscScalar *x, PetscScalar *y)
{
  for (int i = 0; i < dim; i++) {
    y[i] = 0.0;
    for (int j = 0; j < dim; j++) { y[i] += G[i + j * dim] * x[j]; }
  }
}

static inline PetscScalar TinyDot(PetscInt dim, const PetscScalar *x, const PetscScalar *y)
{
  PetscScalar sum = 0.0;
  for (int i = 0; i < dim; i++) sum += x[i] * y[i];
  return sum;
}

static PetscErrorCode KSPSetUp_CACG(KSP ksp)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;
  if (cg->s < 1) SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "s-step size must be >= 1");

  if (!KSP_CACG_BasisGen) {
    PetscCall(PetscLogEventRegister("CACG BasisGen", KSP_CLASSID, &KSP_CACG_BasisGen));
    PetscCall(PetscLogEventRegister("CACG GramCalc", KSP_CLASSID, &KSP_CACG_Gram));
    PetscCall(PetscLogEventRegister("CACG InnerLoop", KSP_CLASSID, &KSP_CACG_InnerLoop));
    PetscCall(PetscLogEventRegister("CACG Reconstruct", KSP_CLASSID, &KSP_CACG_Reconstruct));
  }

  PetscInt s   = cg->s;
  PetscInt dim = 2 * s + 1;
  PetscInt m_local, M_global;
  Mat      A;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatGetLocalSize(A, &m_local, NULL));
  PetscCall(MatGetSize(A, &M_global, NULL));

  if (!cg->V) { PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp), m_local, PETSC_DECIDE, M_global, dim, NULL, &cg->V)); }

  if (!cg->p) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->p));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->r));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->mon_tmp));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &cg->t_vec));
  }
  if (!cg->v_tmp) { PetscCall(MatCreateVecs(A, &cg->v_tmp, NULL)); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_CACG(KSP ksp)
{
  KSP_CACG *cg = (KSP_CACG *)ksp->data;
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
  PetscCall(VecAYPX(r, -1.0, b));
  PetscCall(VecCopy(r, p));

  PetscCall(VecNorm(r, NORM_2, &r_norm));
  ksp->rnorm = r_norm;
  ksp->its   = 0;

  PetscCall(KSPMonitor(ksp, 0, r_norm));
  PetscCall((*ksp->converged)(ksp, 0, r_norm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  while (ksp->its < ksp->max_it && !ksp->reason) {
    // --- EVENT: BASIS GENERATION ---
    // P-Basis
    PetscCall(PetscLogEventBegin(KSP_CACG_BasisGen, ksp, 0, 0, 0));
    {
      Vec col_vec;
      PetscCall(MatDenseGetColumnVecWrite(cg->V, 0, &col_vec));
      PetscCall(VecCopy(p, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(cg->V, 0, &col_vec));

      for (i = 0; i < s; i++) {
        Vec p_prev, p_next;
        PetscCall(MatDenseGetColumnVecRead(cg->V, i, &p_prev));
        PetscCall(KSP_MatMult(ksp, A, p_prev, cg->v_tmp));
        PetscCall(MatDenseRestoreColumnVecRead(cg->V, i, &p_prev));

        PetscCall(MatDenseGetColumnVecWrite(cg->V, i + 1, &p_next));
        PetscCall(VecCopy(cg->v_tmp, p_next));
        PetscCall(MatDenseRestoreColumnVecWrite(cg->V, i + 1, &p_next));
      }
    }
    // R-Basis
    {
      Vec col_vec;
      PetscCall(MatDenseGetColumnVecWrite(cg->V, s + 1, &col_vec));
      PetscCall(VecCopy(r, col_vec));
      PetscCall(MatDenseRestoreColumnVecWrite(cg->V, s + 1, &col_vec));

      for (i = 0; i < s - 1; i++) {
        Vec r_prev, r_next;
        PetscCall(MatDenseGetColumnVecRead(cg->V, s + 1 + i, &r_prev));
        PetscCall(KSP_MatMult(ksp, A, r_prev, cg->v_tmp));
        PetscCall(MatDenseRestoreColumnVecRead(cg->V, s + 1 + i, &r_prev));

        PetscCall(MatDenseGetColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));
        PetscCall(VecCopy(cg->v_tmp, r_next));
        PetscCall(MatDenseRestoreColumnVecWrite(cg->V, s + 1 + i + 1, &r_next));
      }
    }
    PetscCall(PetscLogEventEnd(KSP_CACG_BasisGen, ksp, 0, 0, 0));

    // --- EVENT: OPTIMIZED GRAM MATRIX CALC ---
    PetscCall(PetscLogEventBegin(KSP_CACG_Gram, ksp, 0, 0, 0));

    PetscScalar *v_arr;
    PetscInt     m_local, dim = 2 * s + 1;
    PetscCall(MatDenseGetArrayRead(cg->V, (const PetscScalar **)&v_arr));
    PetscCall(MatGetLocalSize(cg->V, &m_local, NULL));

    PetscScalar *g_local, *g_global;
    PetscCall(PetscCalloc2(dim * dim, &g_local, dim * dim, &g_global));

    PetscBLASInt m_b, n_b, k_b;
    PetscCall(PetscBLASIntCast(dim, &m_b));
    PetscCall(PetscBLASIntCast(dim, &n_b));
    PetscCall(PetscBLASIntCast(m_local, &k_b));

    PetscScalar alpha_blas = 1.0, beta_blas = 0.0;

    PetscCallBLAS("BLASgemm", BLASgemm_("T", "N", &m_b, &n_b, &k_b, &alpha_blas, v_arr, &k_b, v_arr, &k_b, &beta_blas, g_local, &m_b));

    PetscCall(MatDenseRestoreArrayRead(cg->V, (const PetscScalar **)&v_arr));

    PetscCall(MPIU_Allreduce(g_local, g_global, dim * dim, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)ksp)));

    if (!cg->G_repl) { PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, dim, dim, NULL, &cg->G_repl)); }

    PetscScalar *mat_ptr;
    PetscCall(MatDenseGetArrayWrite(cg->G_repl, &mat_ptr));
    PetscCall(PetscArraycpy(mat_ptr, g_global, dim * dim));
    PetscCall(MatDenseRestoreArrayWrite(cg->G_repl, &mat_ptr));

    PetscCall(MatAssemblyBegin(cg->G_repl, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(cg->G_repl, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscFree2(g_local, g_global));
    PetscCall(PetscLogEventEnd(KSP_CACG_Gram, ksp, 0, 0, 0));

    // --- EVENT: INNER LOOP (POINTER OPTIMIZED) ---
    PetscCall(PetscLogEventBegin(KSP_CACG_InnerLoop, ksp, 0, 0, 0));

    // Get pointers once
    PetscScalar *ptr_x, *ptr_p, *ptr_r, *ptr_G, *ptr_mon, *ptr_t;
    PetscCall(VecSet(cg->x, 0.0));
    PetscCall(VecSet(cg->p, 0.0));
    PetscCall(VecSetValue(cg->p, 0, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(cg->p));
    PetscCall(VecAssemblyEnd(cg->p));
    PetscCall(VecSet(cg->r, 0.0));
    PetscCall(VecSetValue(cg->r, s + 1, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(cg->r));
    PetscCall(VecAssemblyEnd(cg->r));

    PetscCall(VecGetArray(cg->x, &ptr_x));
    PetscCall(VecGetArray(cg->p, &ptr_p));
    PetscCall(VecGetArray(cg->r, &ptr_r));
    PetscCall(VecGetArray(cg->mon_tmp, &ptr_mon));
    PetscCall(VecGetArray(cg->t_vec, &ptr_t));
    PetscCall(MatDenseGetArray(cg->G_repl, &ptr_G));

    for (int j = 0; j < s && ksp->its < ksp->max_it; j++) {
      ksp->its++;
      PetscScalar delta, gamma, alpha, beta;

      // 1. delta = r^T * G * r
      TinyMatMult(dim, ptr_G, ptr_r, ptr_mon); // mon = G*r
      delta = TinyDot(dim, ptr_r, ptr_mon);

      // 2. shift P and calc gamma = p^T * G * shift(p)
      // shift p (mon_tmp stores shift(p))
      for (int k = 0; k < dim; k++) ptr_mon[k] = 0.0;
      for (int k = 0; k < s; k++) ptr_mon[k + 1] += ptr_p[k];
      for (int k = 0; k < s - 1; k++) ptr_mon[s + 1 + k + 1] += ptr_p[s + 1 + k];

      TinyMatMult(dim, ptr_G, ptr_mon, ptr_t); // t = G * shift(p)
      gamma = TinyDot(dim, ptr_p, ptr_t);

      if (PetscAbsScalar(gamma) == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }

      alpha = delta / gamma;

      // 3. Update x, p, r
      // x = x + alpha * p
      for (int k = 0; k < dim; k++) ptr_x[k] += alpha * ptr_p[k];
      // r = r - alpha * shift(p) (ptr_mon currently holds shift(p))
      for (int k = 0; k < dim; k++) ptr_r[k] -= alpha * ptr_mon[k];

      // 4. beta
      TinyMatMult(dim, ptr_G, ptr_r, ptr_t); // t = G*r
      PetscScalar delta_new = TinyDot(dim, ptr_r, ptr_t);
      beta                  = delta_new / delta;

      // 5. p = r + beta * p
      for (int k = 0; k < dim; k++) ptr_p[k] = ptr_r[k] + beta * ptr_p[k];

      r_norm = PetscSqrtReal(PetscRealPart(delta_new));
      PetscCall(KSPMonitor(ksp, ksp->its, r_norm));

      if (r_norm < ksp->rtol) {
        ksp->reason = KSP_CONVERGED_RTOL;
        break;
      }
    }

    // Restore pointers
    PetscCall(VecRestoreArray(cg->x, &ptr_x));
    PetscCall(VecRestoreArray(cg->p, &ptr_p));
    PetscCall(VecRestoreArray(cg->r, &ptr_r));
    PetscCall(VecRestoreArray(cg->mon_tmp, &ptr_mon));
    PetscCall(VecRestoreArray(cg->t_vec, &ptr_t));
    PetscCall(MatDenseRestoreArray(cg->G_repl, &ptr_G));

    PetscCall(PetscLogEventEnd(KSP_CACG_InnerLoop, ksp, 0, 0, 0));

    // --- EVENT: RECONSTRUCTION ---
    PetscCall(PetscLogEventBegin(KSP_CACG_Reconstruct, ksp, 0, 0, 0));

    Mat          V_local;
    Vec          x_local_wrapper, p_local_wrapper, r_local_wrapper;
    PetscScalar *x_ptr, *p_ptr, *r_ptr;
    PetscInt     m_local_rows;

    PetscCall(MatDenseGetLocalMatrix(cg->V, &V_local));
    PetscCall(MatGetLocalSize(cg->V, &m_local_rows, NULL));

    PetscCall(VecGetArray(x, &x_ptr));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, x_ptr, &x_local_wrapper));
    PetscCall(MatMultAdd(V_local, cg->x, x_local_wrapper, x_local_wrapper));
    PetscCall(VecDestroy(&x_local_wrapper));
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(VecGetArray(p, &p_ptr));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, p_ptr, &p_local_wrapper));
    PetscCall(MatMult(V_local, cg->p, p_local_wrapper));
    PetscCall(VecDestroy(&p_local_wrapper));
    PetscCall(VecRestoreArray(p, &p_ptr));

    PetscCall(VecGetArray(r, &r_ptr));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m_local_rows, r_ptr, &r_local_wrapper));
    PetscCall(MatMult(V_local, cg->r, r_local_wrapper));
    PetscCall(VecDestroy(&r_local_wrapper));
    PetscCall(VecRestoreArray(r, &r_ptr));

    PetscCall(PetscLogEventEnd(KSP_CACG_Reconstruct, ksp, 0, 0, 0));

    PetscCall(VecNorm(r, NORM_2, &r_norm));
    ksp->rnorm = r_norm;
    if (!ksp->reason) { PetscCall((*ksp->converged)(ksp, ksp->its, r_norm, &ksp->reason, ksp->cnvP)); }
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
  PetscCall(VecDestroy(&cg->p));
  PetscCall(VecDestroy(&cg->r));
  PetscCall(VecDestroy(&cg->x));
  PetscCall(VecDestroy(&cg->mon_tmp));
  PetscCall(VecDestroy(&cg->t_vec));
  PetscCall(MatDestroy(&cg->V));
  PetscCall(MatDestroy(&cg->G_repl));
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

PETSC_EXTERN PetscErrorCode KSPCreate_CACG(KSP ksp)
{
  KSP_CACG *cg;
  PetscNew(&cg);
  ksp->data                = (void *)cg;
  cg->s                    = 1;
  ksp->ops->setup          = KSPSetUp_CACG;
  ksp->ops->solve          = KSPSolve_CACG;
  ksp->ops->destroy        = KSPDestroy_CACG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CACG;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  return 0;
}

int main(int argc, char **args)
{
  Vec           x, b, u;
  Mat           A;
  KSP           ksp;
  char          file[PETSC_MAX_PATH_LEN];
  PetscBool     flg;
  PetscInt      i, n = 100, col[3];
  PetscScalar   value[3];
  PetscReal     norm;
  PetscViewer   viewer;
  PetscLogStage stage_cg, stage_cacg;

  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscCall(KSPRegister(KSPCACG, KSPCreate_CACG));

  // --- 1. PRE-LOAD/GENERATE MATRIX ---
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
    // GENERATING LAPLACIAN
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

  // Setup RHS (u=1, b=A*u)
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  PetscCall(PetscLogStageRegister("Solve_CG", &stage_cg));
  PetscCall(PetscLogStageRegister("Solve_CACG", &stage_cacg));

  // --- 2. SOLVE STANDARD CG ---
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> STARTING STANDARD CG <<<\n"));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetType(ksp, KSPCG));

  PetscCall(PetscLogStagePush(stage_cg));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(PetscLogStagePop());

  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &i));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CG RESULT: Iters %" PetscInt_FMT ", Err %g\n", i, (double)norm));
  PetscCall(KSPDestroy(&ksp));

  // --- 3. SOLVE CACG ---
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> STARTING CACG <<<\n"));
  PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCACG));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PetscLogStagePush(stage_cacg));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(PetscLogStagePop());

  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &i));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CACG RESULT: Iters %" PetscInt_FMT ", Err %g\n", i, (double)norm));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
