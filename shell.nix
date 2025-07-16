{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Build inputs for MPI and PETSc
  buildInputs = with pkgs; [
    openmpi
    petsc
    gcc
    pkg-config
    openssh
  ];

  # A shell hook to provide some helpful information when entering the shell
  shellHook = ''
    echo "Welcome to a Nix shell with MPI and PETSc!"
    echo "Using MPI implementation: ${pkgs.openmpi.name}"
    echo "Using PETSc version: ${pkgs.petsc.version}"
    echo ""
    echo "You can compile MPI programs using 'mpicc', 'mpicxx', 'mpif90', etc."
    echo "PETSc libraries and headers should be available in your path."
    echo ""
  '';
}