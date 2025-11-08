# conjugate gradient solver

compile with:
```
mpicc <filename>.c -o cg $(pkg-config --cflags --libs PETSc)
```

run with:
```
mpirun -n 1 --oversubscribe cg
```

formatted using `.clang-format` file provided by [PETSc](https://gitlab.com/petsc/petsc)
