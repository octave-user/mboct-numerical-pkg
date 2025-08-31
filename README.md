# mboct-numerical-pkg<sup>&copy;</sup>
**mboct-numerical-pkg** belongs to a suite of packages which can be used for pre- and postprocessing of MBDyn models (https://www.mbdyn.org) with GNU-Octave (http://www.gnu.org/software/octave/). This package contains interfaces to several well known numerical solvers.

# List of features
  - Solve large scale sparse symmetric linear systems of equations using PaStiX (https://gitlab.inria.fr/solverstack/pastix), MUMPS (https://github.com/group-gu/mumps) or UMFPACK (http://faculty.cse.tamu.edu/davis/suitesparse.html).
  - Solve large scale generalized symmetric eigenvalue problems with ARPACK (https://www.caam.rice.edu/software/ARPACK/).
  - Compute fill in reducing node orderings with METIS (http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).
  - Compute sparse matrix vector products for symmetric sparse matrices where only the upper or lower triangular part of the matrix is stored.
  - Compute selected eigenvalues, and optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem using LAPACK (http://www.netlib.org/lapack/).
  - Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage using LAPACK.

Copyright<sup>&copy;</sup> 2019-2025

[Reinhard](mailto:octave-user@a1.net)

# Installation
  - See [simple.yml](https://github.com/octave-user/mboct-numerical-pkg/blob/master/.github/workflows/simple.yml) as an example on how to install mboct-numerical-pkg.

# Function reference
  - The function reference is automatically generated from the source code by means of Octave's [generate_html](https://octave.sourceforge.io/generate_html/index.html) package. See [overview.html](https://octave-user.github.io/mboct-numerical-pkg/mboct-numerical-pkg/overview.html).

## Usage
  - Run Octave.
    `octave`
  - At the Octave prompt load the package.
    `pkg load mboct-numerical-pkg`
  - At the Octave prompt execute the tests.
    `test numerical_tests_01.tst`
