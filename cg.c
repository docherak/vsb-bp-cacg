#include <petscksp.h>

static char help[] = "Conjugate gradient solver.\n\n";

int main(int argc, char **args) {
    Mat         A;
    Vec         x, b;
    PetscInt    M = 3;
    PetscInt    i, j;
    PetscScalar v, tol;
    PetscInt    rowStart, rowEnd, rank, size;
    PetscErrorCode ierr;
    PetscBool       flg;
    PetscLogDouble  t1, t2;


    ierr = PetscInitialize(&argc, &args, (char *)0, help);CHKERRQ(ierr);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    ierr = PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-tol",   &tol,   &flg); CHKERRQ(ierr);

    /* Create the matrix */
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);

    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, M);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);

    /* Indicate that the matrix is symmetric positive definite */
    ierr = MatSetOption(A, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);

    /* Populate the matrix (HARDCODED SPD matrix) */
    ierr = MatGetOwnershipRange(A, &rowStart, &rowEnd);CHKERRQ(ierr);

    // Hardcode a 3x3 SPD matrix as an example.
    // Example SPD Matrix:
    // [ 4 -1  0 ]
    // [-1  4 -1 ]
    // [ 0 -1  4 ]

    // Row 0
    i = 0;
    if (i >= rowStart && i < rowEnd) {
        j = 0; v = 4.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
        j = 1; v = -1.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    // Row 1
    i = 1;
    if (i >= rowStart && i < rowEnd) {
        j = 0; v = -1.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
        j = 1; v = 4.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
        j = 2; v = -1.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    // Row 2
    i = 2;
    if (i >= rowStart && i < rowEnd) {
        j = 1; v = -1.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
        j = 2; v = 4.0; ierr = MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }

    /* Assemble the matrix */
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Get the standard output viewer (using the correct function)
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);

    // Push the dense format onto the viewer
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);CHKERRQ(ierr);

    /* View the matrix in dense format */
    PetscPrintf(PETSC_COMM_WORLD, "--- Matrix A (Dense Format) ---\n");
    ierr = MatView(A, viewer);CHKERRQ(ierr);

    // Pop the dense format (return to default)
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);

    /* Create vectors for testing (e.g., for A*x) */
    ierr = MatCreateVecs(A, &x, &b);CHKERRQ(ierr);
    ierr = VecSet(x, 2.0);CHKERRQ(ierr); // Example: x = [2, 2, ..., 2]^T (adjusted for consistency)
    ierr = MatMult(A, x, b);CHKERRQ(ierr); // b = A*x

    ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* CG Solver */

    ierr = PetscTime(&t1); 	    CHKERRQ(ierr);

    Vec         x_k, r_k, d_k, Ad_k;      /* vectors */
    PetscScalar r_norm, r_k_dot_r_k, r_k1_dot_r_k1, d_k_dot_Ad_k, alpha_k, beta_k;
    PetscInt max_iter = 1000;
    PetscInt iter;

    ierr = VecDuplicate(b, &x_k);CHKERRQ(ierr); // x1 is your solution vector
    ierr = VecDuplicate(b, &r_k);CHKERRQ(ierr);  // r is the residual vector
    ierr = VecDuplicate(b, &d_k);CHKERRQ(ierr);  // d is the search direction vector
    ierr = VecDuplicate(b, &Ad_k);CHKERRQ(ierr); // Ad is A*d

    // 1. Initialize x_0 = [0, 0, ..., 0]
    ierr = VecSet(x_k, 0.0);CHKERRQ(ierr);

    // 2. Compute initial residual r_0 = b - A*x_0
    // Since x_0 is 0, r_0 = b
    ierr = VecCopy(b, r_k);CHKERRQ(ierr);

    // 3. Set initial search direction p_0 = r_0
    ierr = VecCopy(r_k, d_k);CHKERRQ(ierr);

    // 4. Compute initial residual norm (for convergence check)
    ierr = VecNorm(r_k, NORM_2, &r_norm);CHKERRQ(ierr);

    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "\nStarting CG solver with tolerance %g\n", (double)tol);
    }

    for (iter = 0; iter < max_iter && r_norm > tol; iter++) {
        // a. Compute r_k_dot_r_k = r_k^T * r_k (numerator for alpha)
        ierr = VecDot(r_k, r_k, &r_k_dot_r_k);CHKERRQ(ierr);
        // b. Compute Ad_k = A * d_k
        ierr = MatMult(A, d_k, Ad_k);CHKERRQ(ierr);
        // c. Compute d_k_dot_Ad_k = d_k^T * Ad_k
        ierr = VecDot(d_k, Ad_k, &d_k_dot_Ad_k);CHKERRQ(ierr);
        // d. Compute alpha_k
        alpha_k = r_k_dot_r_k / d_k_dot_Ad_k;
        // e. Compute x_{k+1}
        ierr = VecAXPY(x_k, alpha_k, d_k);CHKERRQ(ierr);
        // f. Compute r_{k+1}
        ierr = VecAXPY(r_k, -alpha_k, Ad_k);CHKERRQ(ierr);
        // g. Compute r_k1_dot_r_k1 = r_{k+1}^T * r_{k+1} (numerator for beta)
        ierr = VecDot(r_k, r_k, &r_k1_dot_r_k1);CHKERRQ(ierr);
        // h. Compute beta_k
        beta_k = r_k1_dot_r_k1 / r_k_dot_r_k;
        // i. Compute d_{k+1} = r_{k+1} + beta_k * d_k
        ierr = VecScale(d_k, beta_k);CHKERRQ(ierr);
        ierr = VecAXPY(d_k, 1.0, r_k);CHKERRQ(ierr);

        ierr = VecNorm(r_k, NORM_2, &r_norm);CHKERRQ(ierr);
    }

    if (rank == 0) {
        if (r_norm <= tol) {
            PetscPrintf(PETSC_COMM_WORLD, "CG converged in %D iterations.\n", iter);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "CG did NOT converge in %D iterations. Final residual norm: %g\n", iter, (double)r_norm);
        }
    }

    ierr = PetscTime(&t2); 	    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "\n--- Final Solution Vector (x_k) ---\n");
    PetscPrintf(PETSC_COMM_WORLD,"Time of solution : %5.3e\n", t2-t1);
    ierr = VecView(x_k, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* Destroy PETSc objects */
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&x_k);CHKERRQ(ierr);
    ierr = VecDestroy(&r_k);CHKERRQ(ierr);
    ierr = VecDestroy(&d_k);CHKERRQ(ierr);
    ierr = VecDestroy(&Ad_k);CHKERRQ(ierr);

    ierr = PetscFinalize();CHKERRQ(ierr);
    return 0;
}
