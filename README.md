# mboct-numerical-pkg<sup>&copy;</sup>
**mboct-numerical-pkg** belongs to a suite of packages which can be used for pre- and postprocessing of MBDyn models (https://www.mbdyn.org) with GNU-Octave (http://www.gnu.org/software/octave/). This package contains interfaces to several well known numerical solvers.

# List of features
  - Solve large scale sparse symmetric linear systems of equations using PaStiX (https://gitlab.inria.fr/solverstack/pastix), MUMPS (https://github.com/group-gu/mumps) or UMFPACK (http://faculty.cse.tamu.edu/davis/suitesparse.html).
  - Solve large scale generalized symmetric eigenvalue problems with ARPACK (https://www.caam.rice.edu/software/ARPACK/).
  - Compute fill in reducing node orderings with METIS (http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).
  - Compute sparse matrix vector products for symmetric sparse matrices where only the upper or lower triangular part of the matrix is stored.
  - Compute selected eigenvalues, and optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem using LAPACK (http://www.netlib.org/lapack/).
  - Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage using LAPACK.
  
Copyright<sup>&copy;</sup> 2019-2021

[Reinhard](mailto:octave-user@a1.net)

# Installation
  The following code is an example how mboct-numerical-pkg can be installed on an Ubuntu system:
  
  `sudo apt-get install octave liboctave-dev libsuitesparse-dev libarpack2-dev libmumps-seq-dev libmetis-dev libmkl-full-dev`

  `git clone -b master https://github.com/octave-user/mboct-numerical-pkg.git`
       
  `make -C mboct-numerical-pkg install_local`

## PaStiX installation
  - Follow the instructions on (https://gitlab.inria.fr/solverstack/pastix) to install PaStiX (optional but recommended).

## MUMPS installation
  - Follow the instructions on (https://github.com/group-gu/mumps) to install MUMPS (optional).

## GNU Octave installation
  - Follow the instructions on (http://www.gnu.org/software/octave/) to install GNU Octave.  
  - Make sure, that `mkoctfile` is installed.  
    `mkoctfile --version` 

## GNU Octave package installation:
  - Install the following packages from github.  
    `for pkg in numerical; do`    
        `git clone https://github.com/octave-user/mboct-${pkg}-pkg.git && make -C mboct-${pkg}-pkg install_local`	  
    `done`

## Usage
  - Run Octave.  
    `octave`
  - At the Octave prompt load the package.   
    `pkg load mboct-numerical-pkg`
  - At the Octave prompt execute the tests.  
    `test numerical_tests`
	
