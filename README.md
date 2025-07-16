# conjugate gradient solver

compile with:
```
mpicc cg.c -o cg $(pkg-config --cflags --libs PETSc)
```

run with:
```
mpirun -n 1 cg
```