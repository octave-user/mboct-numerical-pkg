// Copyright (C) 2018(-2021) Reinhard <octave-user@a1.net>

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

// PKG_ADD: autoload ("sp_sym_mtimes", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("sp_sym_mtimes", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (sp_sym_mtimes, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{b} = sp_sym_mtimes(@var{A},  @var{x})\n\n"
           "Compute the product @var{b} = (@var{A} + @var{A}.' - diag(diag(@var{A})))  * @var{x}\n\n"
	   "@var{A} @dots{} upper- or lower-triangular part of a real symmetric sparse matrix\n\n"
	   "@var{x} @dots{} a real dense matrix\n\n"
	   "@var{b} @dots{} the matrix - matrix product\n"
           "@end deftypefn\n")
{
        octave_value_list retval;

        if (args.length() != 2 || nargout > 1) {
                print_usage();
                return retval;
        }

        if (!args(0).is_matrix_type() && args(1).is_matrix_type()) {
                error("A and x must be matrices");
                return retval;
        }

        if (args(0).rows() != args(0).columns()) {
                error("Matrix A is not square");
                return retval;
        }

        if (args(0).columns() != args(1).rows()) {
                error("Number of columns of A does not match number of rows of x");
                return retval;
        }

        const SparseMatrix A = args(0).sparse_matrix_value();
        const Matrix x = args(1).matrix_value();

        const octave_idx_type* const Aridx = A.ridx();
        const octave_idx_type* const Acidx = A.cidx();
        const double* const Adata = A.data();

        Matrix b(A.rows(), x.columns(), 0.);

        for (octave_idx_type l = 0; l < x.columns(); ++l) {
                for (octave_idx_type j = 0; j < A.columns(); ++j) {
                        for (octave_idx_type k = Acidx[j]; k < Acidx[j + 1]; ++k) {
                                const octave_idx_type i = Aridx[k];

                                const double Aij = Adata[k];

                                b.xelem(i, l) += Aij * x.xelem(j, l);

                                if (i != j) {
                                        b.xelem(j, l) += Aij * x.xelem(i, l);
                                }

                        }
                }
        }

        retval.append(b);

        return retval;
}

/*
%!test
%! state = rand("state");
%! unwind_protect
%!   rand("seed", 0);
%!   tol = eps^0.9;
%!   for i=1:10
%!     for N=[1, 10, 100, 1000]
%!       for M=[1, 10, 20, 100]
%!         A = sprand(N, N, 0.01);
%!         A += A.';
%!         x = rand(N, M);
%!         b = A * x;
%!         [r, c, d] = find(A);
%!         for k=1:2
%!           switch (k)
%!           case 1
%!             idx = find(r >= c);
%!           case 2
%!             idx = find(r <= c);
%!           endswitch
%!           A2 = sparse(r(idx), c(idx), d(idx), N, N);
%!           b2 = sp_sym_mtimes(A2, x);
%!           assert(b2, b, tol * max(max(abs(b))));
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! unwind_protect_cleanup
%!   rand("state", state);
%! end_unwind_protect
*/
