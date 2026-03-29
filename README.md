# CACG PETSc KSP Solver

This project is part of the **Parallel implementation of the $s$-step conjugate
gradient method** bachelor thesis, which covers the state of contemporary research regarding $s$-step conjugate gradient method and focuses on implementation of this algorithm in [PETSc](https://petsc.org) and the subsequent experiments in  HPC (High-Performance Computing) environment.

It contains:
- custom PETSc implementations of various versions of (standard) conjugate gradient (CG) method written as part of a the research phase (`p<N>_<paper_name>` folders);
- custom naive implementations of the $s$-step method (`iterations` folder);
- custom more optimized implementation of $s$-step CACG as [KSP solver](https://petsc.org/main/manual/ksp/), labelled `KSPCACG` (`ksp_cacg.c` file in root);
- (TODO) packaged results from numerical experiments in [LUMI-C](https://docs.lumi-supercomputer.eu/hardware/lumic/) HPC environment.


Final code is formatted using `.clang-format` file provided by [PETSc GitLab](https://gitlab.com/petsc/petsc) repository.

## OLD:
compile with:
```
mpicc <filename>.c -o cg $(pkg-config --cflags --libs PETSc)
```

run with:
```
mpirun -n 1 --oversubscribe cg
```

