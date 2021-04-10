// Copyright (C) 2019(-2021) Reinhard <octave-user@a1.net>

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

#include <array>
#include <iostream>
#include <octave/oct.h>
#include <octave/interpreter.h>

#define HAVE_ARPACK 1
#include <octave/lo-arpack-proto.h>

// PKG_ADD: autoload ("eig_sym", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("eig_sym", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (eig_sym, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} [@var{v}, @var{lambda}] = eig_sym(@var{OP}, @var{N}, @var{NEV}, @var{SIGMA}, @var{OPT})\n"
           "Solve the general real symmetric eigenvalue problem @var{A} * @var{v} = @var{lambda} * @var{B} * @var{v} "
           "by using ARPACK's dsaupd and dseupd subroutines.\n"
           "Matrices @var{A} and @var{B} must be symmetric but in general they need not be definite or regular,\n"
           "provided that the matrix operations @var{OP} can be defined.\n\n"
           "@var{OP} @dots{} Cell array of function handles or inline functions for matrix operations.\n\n"
           "If @var{SIGMA} is equal to \"SM\" or \"LM\" then\n\n"
           "@var{OP}@{1@}(@var{x}) must return @var{A} * @var{x}\n\n"
           "@var{OP}@{2@}(@var{x}) must return @var{B} \\ @var{x}\n\n"
           "@var{OP}@{3@}(@var{x}) must return @var{B} * @var{x}\n\n"
           "If @var{SIGMA} is a real scalar then\n\n"
           "@var{OP}@{1@}(@var{x}) must return @var{B} * @var{x}\n\n"
           "@var{OP}@{2@}(@var{x}) must return (@var{A} - @var{SIGMA} * @var{B}) \\ @var{x}\n\n"
           "@var{SIGMA} @dots{} If @var{SIGMA} is a real scalar, find @var{OPT}.p eigenvalues close to @var{SIGMA}.\n\n"
           "If @var{SIGMA} is equal to \"LM\" or \"SM\", find @var{OPT}.p eigenvalues of largest or smallest magnitude respectively.\n\n"
           "@var{N} @dots{} The order of matrix @var{A}.\n\n"
           "@var{NEV} @dots{} The number of eigenvalues requested.\n\n"
           "@var{OPT}.maxit @dots{} Optional maximum number of iterations for ARPACK.\n\n"
           "@var{OPT}.tol @dots{} Optional tolerance for ARPACK.\n\n"
           "@var{OPT}.v0 @dots{} Optional starting vector for ARPACK.\n\n"
           "@var{OPT}.p @dots{} Optional number of Arnoldi vectors to use for ARPACK.\n\n"
           "@end deftypefn\n")
{
    octave_value_list retval;

    if (args.length() != 5) {
        print_usage();
        return retval;
    }

    Cell ov_op = args(0).cell_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif

    for (octave_idx_type i = 0; i < ov_op.numel(); ++i) {       
      if (!(ov_op(i).is_function() ||
	    ov_op(i).is_function_handle() ||
	    ov_op(i).is_anonymous_function() ||
	    ov_op(i).is_inline_function())) {
	    error("argument OP must be a cell array of functions");
            return retval;
        }
    }

    const F77_INT n = args(1).int_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif

    const F77_INT nev = args(2).int_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif
    
    std::string type;
    double sigma = 0.;
    F77_INT mode = -1;
    
    if (args(3).is_string() || args(3).is_sq_string()) {
        type = args(3).string_value();

#if OCTAVE_MAJOR_VERSION < 6
        if (error_state) {
            return retval;
        }
#endif
        
        mode = 2;

        if (ov_op.numel() != 3) {
            error("eig_sym: invalid number of operations provided");
            return retval;
        }
    } else if (args(3).is_real_scalar()) {
        sigma = args(3).scalar_value();

#if OCTAVE_MAJOR_VERSION < 6
        if (error_state) {
            return retval;
        }
#endif

        type = "LM";
        mode = 3;

        if (ov_op.numel() != 2) {
            error("eig_sym: invalid number of operations provided");
        }
    } else {
        error("eig_sym: invalid value for SIGMA");
        return retval;
    }
    
    if (type != "SM" && type != "LM") {
        error("invalid value for TYPE");
        return retval;
    }

    const octave_scalar_map opt = args(4).scalar_map_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif
    
    const auto itp = opt.seek("p");

    const F77_INT ncv = itp == opt.end() ? 2 * nev : opt.contents(itp).int_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif
    
    const auto itmaxit = opt.seek("maxit");

    const octave_idx_type maxit = itmaxit == opt.end() ? 300 : opt.contents(itmaxit).int_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif
    
    const auto ittol = opt.seek("tol");

    const F77_DBLE tol = ittol == opt.end() ? 0. : opt.contents(ittol).scalar_value();

    const auto itv0 = opt.seek("v0");

    F77_INT info;
    ColumnVector resid;

    if (itv0 != opt.end()) {
        resid = opt.contents(itv0).column_vector_value();

#if OCTAVE_MAJOR_VERSION < 6
        if (error_state) {
            return retval;
        }
#endif
        
        if (resid.rows() != n) {
            error("eig_sym: invalid number of rows for opt.v0");
            return retval;
        }
        
        info = 1;
    } else {
        resid.resize(n);
        info = 0;
    }
    
    static const char bmat[] = "G";

    Array<F77_INT> ip (dim_vector(11, 1));
    F77_INT *const iparam = ip.fortran_vec();

    constexpr F77_INT ishfts = 1;

    ip(0) = ishfts;
    ip(2) = maxit;
    ip(6) = mode;

    Array<F77_INT> iptr(dim_vector (11, 1));
    F77_INT *const ipntr = iptr.fortran_vec();

    F77_INT ido = 0;
    const F77_INT lwork = ncv * (ncv + 8);

    Matrix eig_vec(n, ncv);
    double *const v = eig_vec.fortran_vec();

    RowVector workl(lwork);
    double* const worklp = workl.fortran_vec();
    RowVector workd(3 * n);
    double* const workdp = workd.fortran_vec();    
    double *const presid = resid.fortran_vec();
    ColumnVector w(n);

    do {
        F77_FUNC (dsaupd, DSAUPD)
            (ido,
             F77_CONST_CHAR_ARG2 (bmat, 1),
             n,
             F77_CONST_CHAR_ARG2 (type.c_str(), 2),
             nev,
             tol,
             presid,
             ncv,
             v,
             n,
             iparam,
             ipntr,
             workdp,
             worklp,
             lwork,
             info
             F77_CHAR_ARG_LEN(1)
             F77_CHAR_ARG_LEN(2));

        OCTAVE_QUIT;
        
#if EIGSYM_TRACE > 0
        std::cout << "ido=" << ido << std::endl;
        std::cout << "bmat=" << bmat << std::endl;
        std::cout << "type=" << type << std::endl;
        std::cout << "resid=" << resid << std::endl;
        std::cout << "info=" << info << std::endl;
        std::cout << "v=" << eig_vec << std::endl;
        std::cout << "workd=" << workd << std::endl;
        std::cout << "workl=" << workl << std::endl;
        std::cout << "iparam=";

        for (octave_idx_type i = 0; i < ip.numel(); ++i) {
            std::cout << ip(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "iptr=";
        for (octave_idx_type i = 0; i < iptr.numel(); ++i) {
            std::cout << iptr(i) << " ";
        }
        std::cout << std::endl;
#endif

        if (ido == -1 || ido == 1 || ido == 2) {
            F77_INT iinptr;

            if (mode == 2) {
                iinptr = iptr(0) - 1; // w = x
            } else if (ido == -1 || ido == 2) {
                iinptr = iptr(0) - 1; // w = x
            } else {
                iinptr = iptr(2) - 1; // w = B * x
            }
            
            for (F77_INT i = 0; i < n; i++) {
                w(i) = workd(i + iinptr);
            }

            octave_idx_type iop;
            
            if (mode == 2) {
                iop = (ido == 1 || ido == -1)
                    ? 0  // z = A * x
                    : 2; // z = B * x
            } else if (ido == -1 || ido == 2) {                
                iop = 0; // z = B * x
            } else {
                iop = 1; // z = (A - sigma * B)^-1 * w = (A - sigma * B)^-1 * B * x
            }
            
            octave_value_list f = OCTAVE__FEVAL(ov_op(iop), octave_value(w), 1);

#if OCTAVE_MAJOR_VERSION < 6
            if (error_state) {
                return retval;
            }
#endif
            
            if (f.length() != 1) {
                error("eig_sym: invalid number of output arguments");
                return retval;
            }

            ColumnVector y;

            if ((mode == 2 && (ido == 1 || ido == -1)) || (mode == 3 && ido == -1)) {
                const ColumnVector z = f(0).column_vector_value();

#if OCTAVE_MAJOR_VERSION < 6
                if (error_state) {
                    return retval;
                }
#endif
                
                if (z.rows() != n) {
                    error("eig_sym: user supplied function returned invalid number of rows");
                    return retval;
                }

                if (mode == 2) {
                    for (F77_INT i = 0; i < n; i++) {
                        workd(i + iptr(0) - 1) = z(i);
                    }
                }

                // mode == 2 : f = B^-1 * z = B^-1 * A * x
                // mode == 3 : f = (A - sigma * B)^-1 * z = (A - sigma * B)^-1 * B * x
                f = OCTAVE__FEVAL(ov_op(1), octave_value(z), 1);

#if OCTAVE_MAJOR_VERSION < 6
                if (error_state) {
                    return retval;
                }
#endif
                
                if (f.length() != 1) {
                    error("eig_sym: invalid number of output arguments");
                    return retval;
                }                
            }

            y = f(0).column_vector_value();

#if OCTAVE_MAJOR_VERSION < 6
            if (error_state) {
                return retval;
            }
#endif
            
            if (y.rows() != n) {
                error("eig_sym: user supplied function returned an invalid number of rows");
                return retval;
            }

            for (F77_INT i = 0; i < n; i++) {
                workd(i + iptr(1) - 1) = y(i);
            }
#if EIGSYM_TRACE > 0
            std::cout << "workd=" << workd << std::endl;
#endif
        } else {
            break;
        }
    } while (1);

    if (info < 0) {
        error("eig_sym: dsaupd failed with status %d", info);
        return retval;
    } else if (info != 0) {
        warning("eig_sym: dsaupd returned with status %d", info);
    }

    bool rvec = true;
    Array<F77_INT> s(dim_vector(ncv, 1));
    F77_INT *sel = s.fortran_vec();

    ColumnVector eig_val(nev);
    double *d = eig_val.fortran_vec();

    F77_FUNC (dseupd, DSEUPD)
        (rvec,
         F77_CONST_CHAR_ARG2 ("A", 1),
         sel,
         d,
         v,
         n,
         sigma,
         F77_CONST_CHAR_ARG2 (bmat, 1),
         n,
         F77_CONST_CHAR_ARG2 (type.c_str(), 2),
         nev,
         tol,
         presid,
         ncv,
         v,
         n,
         iparam,
         ipntr,
         workdp,
         worklp,
         lwork,
         info
         F77_CHAR_ARG_LEN(1)
         F77_CHAR_ARG_LEN(1)
         F77_CHAR_ARG_LEN(2));

    if (info != 0) {
        error("eig_sym: dseupd failed with status %d", info);
        return retval;
    }

    const F77_INT iconv = iparam[4];

    if (iconv < nev) {
        warning("eig_sym: number of eigenvalues requested: %d, number of eigenvalues converged: %d", nev, iconv);
    }
    
    eig_val.resize(iconv);
    eig_vec.resize(n, iconv);

    retval.append(eig_vec);
    retval.append(DiagMatrix(eig_val));

    return retval;
}

/*
%!function [A,B]=build_test_mat(N)
%! B = gallery("poisson", N);
%! A = gallery("tridiag", columns(B));

%!test
%! rand("seed", 0);
%! for k=[5,10,50,100]
%! [A, B] = build_test_mat(k);
%! for k=1:10
%! opts.v0 = rand(columns(B), 1);
%! sigma = "LM";

%! op{1} = @(x) A * x;
%! op{2} = @(Ax) B \ Ax;
%! op{3} = @(x) B * x;

%! nev = 10;
%! opts.maxit = 3000;
%! opts.p = 20;
%! opts.tol = 0;
%! n = columns(A);
%! [v, lambda] = eig_sym(op, n, nev, sigma, opts);
%! tol = 1e-6;
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(v1, v2, tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! rand("seed", 0);
%! for l=[5,10,20,50,100]
%! [A,B]=build_test_mat(l);
%! for k=1:10
%! opts.v0 = rand(columns(B), 1);
%! sigma = (k - 1) / 1000;
%! op{1} = @(x) B * x;
%! op{2} = @(Bx) (A - sigma * B) \ Bx;
%! nev = 10;
%! opts.maxit = 3000;
%! opts.p = 20;
%! opts.tol = 0;
%! n = columns(A);
%! [v, lambda] = eig_sym(op, n, nev, sigma, opts);
%! tol = 1e-6;
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(v1, v2, tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! trace = false;
%! rand("seed", 0);
%! sigma={"SM","LM"};
%! for s=1:numel(sigma)
%! for n = [10, 20, 50, 100, 200, 500, 1000];
%! nev = 3;
%! ncv = min([n, 2 * nev + floor(5 * sqrt(n))]);
%! h = 1 / (n+1);
%! r1 = (4 / 6) * h;
%! r2 = (1 / 6) * h;
%! B = sparse([],[],[], n, n);
%! for i=1:n
%!   B(i, i) = r1;
%!   if (i + 1 <= n)
%!     B(i, i + 1) = r2;
%!     B(i + 1, i) = r2;
%!   endif
%! endfor

%! A = sparse([], [], [], n, n);
%! A(1, 1) = 2 / h;
%! A(1, 2) = -1 / h;
%! for i=2:n
%!   A(i, i) = 2 / h;
%!   A(i, i - 1) = -1 / h;
%!   A(i - 1, i) = -1 / h;
%! endfor
%! assert(isdefinite(A));
%! assert(isdefinite(B));
%! op{1} = @(x) A * x;
%! op{2} = @(Ax) B \ Ax;
%! op{3} = @(x) B * x;
%! opts.maxit = 300;
%! opts.p = ncv;
%! opts.tol = 0;
%! ## A * x = lambda * B * x
%! v1 = [];
%! lambda1 = [];
%! opts.v0 = rand(n, 1);
%! for k=1:2
%! [v, lambda] = eig_sym(op, n, nev, sigma{s}, opts);
%! if (k == 1)
%! v1 = v;
%! lambda1 = lambda;
%! else
%! assert(v1, v, 0);
%! assert(lambda1, lambda, 0);
%! endif
%! endfor
%! tol = sqrt(eps);
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(v1, v2, tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! trace = false;
%! rand("seed", 0);
%! for n = [10, 20, 50, 100, 200, 500, 1000];
%! nev = 3;
%! ncv = min([n, 2 * nev + floor(5 * sqrt(n))]);
%! h = 1 / (n+1);
%! r1 = (4 / 6) * h;
%! r2 = (1 / 6) * h;
%! B = sparse([],[],[], n, n);
%! for i=1:n
%!   B(i, i) = r1;
%!   if (i + 1 <= n)
%!     B(i, i + 1) = r2;
%!     B(i + 1, i) = r2;
%!   endif
%! endfor

%! A = sparse([], [], [], n, n);
%! A(1, 1) = 2 / h;
%! A(1, 2) = -1 / h;
%! for i=2:n
%!   A(i, i) = 2 / h;
%!   A(i, i - 1) = -1 / h;
%!   A(i - 1, i) = -1 / h;
%! endfor
%! assert(isdefinite(A));
%! assert(isdefinite(B));
%! for sigma = 0:0.1:1;
%! op{1} = @(x) B * x;
%! op{2} = @(Bx) (A - sigma * B) \ Bx;
%! opts.maxit = 300;
%! opts.p = ncv;
%! opts.tol = 0;
%! opts.v0 = rand(n, 1);
%! ## A * x = lambda * B * x
%! for k=1:2
%! [v, lambda] = eig_sym(op, n, nev, sigma, opts);
%! if (k == 1)
%!   v1 = v;
%!   lambda1 = lambda;
%! else
%!   assert(lambda, lambda1, 0);
%!   assert(v, v1, 0);
%! endif
%! endfor
%! tol = sqrt(eps);
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(v1, v2, tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor
 */
