// Copyright (C) 2018(-2023) Reinhard <octave-user@a1.net>

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; If not, see <http://www.gnu.org/licenses/>.

#include "config.h"

#include <octave/oct.h>
#include <octave/f77-fcn.h>

extern "C"
{
    F77_RET_T F77_FUNC(dspev, DSPEV)(F77_CONST_CHAR_ARG_DECL JOBZ,
                                     F77_CONST_CHAR_ARG_DECL UPLO,
                                     F77_INT* N,
                                     F77_DBLE* AP,
                                     F77_DBLE* W,
                                     F77_DBLE* Z,
                                     F77_INT* LDZ,
                                     F77_DBLE* WORK,
                                     F77_INT* INFO);
}

// PKG_ADD: autoload ("dspev", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("dspev", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (dspev, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} [@var{U}, @var{lambda}] = dspev(@var{A})\n"
           "@deftypefnx {} @var{lambda} = dspev(@var{A})\n"
           "Computes all the eigenvalues @var{lambda} and, optionally, eigenvectors @var{U} of a real symmetric matrix @var{A} in packed storage.\n"
           "@end deftypefn\n")
{
    static const char invalid_args[] = "A must be a symmetric sparse matrix";
    octave_value_list retval;

    octave_idx_type nargin = args.length();

    if (nargin != 1 || nargout > 2)
    {
        print_usage();
        return retval;
    }

    if (!args(0).is_matrix_type())
    {
        error(invalid_args);
        return retval;
    }

    const SparseMatrix A = args(0).sparse_matrix_value();

    if (A.rows() != A.cols() || !A.OV_ISSYMMETRIC() || A.numel() == 0)
    {
        error(invalid_args);
        return retval;
    }

    char JOBZ = nargout == 1 ? 'N' : 'V';

    char UPLO = 'U';

    F77_INT N = octave::to_f77_int(A.rows());

    OCTAVE_LOCAL_BUFFER(F77_DBLE, AP, (N * (N + 1) / 2));

    for (octave_idx_type j = 1; j <= N; ++j )
    {
            for (octave_idx_type i = 1; i <= j; ++i )
            {
                    AP[i + (j - 1) * j / 2 - 1] = A(i - 1, j - 1);
            }
    }

    RowVector W(N);

    Matrix Z;

    if (JOBZ == 'V')
    {
            Z.resize(N, N);
    }

    F77_INT LDZ = N;

    OCTAVE_LOCAL_BUFFER(F77_DBLE, WORK, 3 * N);

    F77_INT INFO = 0;

    F77_XFCN(dspev, DSPEV, (&JOBZ, &UPLO, &N, AP, W.fortran_vec(), Z.fortran_vec(), &LDZ, WORK, &INFO));

#if OCTAVE_MAJOR_VERSION < 5
    if ( f77_exception_encountered )
    {
        error("Fortran exception in dspev");
        return retval;
    }
#endif

    if (INFO != 0)
    {
            long lINFO = INFO;

            if (INFO < 0)
            {
                    error("dspev failed with INFO=%ld: the %ld-th argument had an illegal value", lINFO, lINFO);
            }
            else
            {
                    error("dspev failed with INFO=%ld: the algorithm failed to converge, off-diagonal elements of an intermediate tridiagonal form did not converge to zero", lINFO);
            }

            return retval;
    }

    if (nargout == 2)
    {
        retval.append(Z);
    }

    retval.append(W);

    return retval;
}
