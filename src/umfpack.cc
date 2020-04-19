// Copyright (C) 2019(-2020) Reinhard <octave-user@a1.net>

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

#include <stdexcept>
#include <cstring>
#include <octave/oct.h>

#include <suitesparse/umfpack.h>

class UmfpackObject : public octave_base_value {
public:
        struct Options {
                int verbose = UMFPACK_DEFAULT_PRL;
                int refine_max_iter = UMFPACK_DEFAULT_IRSTEP;
        };

        UmfpackObject();
        explicit UmfpackObject(const SparseMatrix& A, const Options& options);
        virtual ~UmfpackObject(void);
        Matrix solve(const Matrix& b);
        virtual bool is_constant(void) const{ return true; }
        virtual bool is_defined(void) const{ return true; }
        virtual dim_vector dims (void) const { return A.dims(); }
        virtual void print(std::ostream& os, bool pr_as_read_syntax) {
        }
private:
        DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA

        void cleanup();

        SparseMatrix A;
        Options options;
        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        void *Symbolic = nullptr;
        void *Numeric = nullptr;        
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(UmfpackObject, "umfpack", "umfpack");

UmfpackObject::UmfpackObject()
{
}

UmfpackObject::UmfpackObject(const SparseMatrix& A, const Options& opt)
        : A(A),
          options(opt)
{
        
        if (A.rows() != A.columns()) {
                throw std::runtime_error("matrix A must be square");
        }

        std::memset(&Info[0], 0, sizeof(Info));
        umfpack_dl_defaults(Control);
        Control[UMFPACK_PRL] = opt.verbose;
        Control[UMFPACK_IRSTEP] = opt.refine_max_iter;
        
        const octave_idx_type n = A.columns();
        const octave_idx_type* const cidx = A.cidx();
        const octave_idx_type* const ridx = A.ridx();
        const double* const data = A.data();

	auto status = umfpack_dl_symbolic(n,
                                          n,
                                          cidx,
                                          ridx,
                                          data,
                                          &Symbolic,
                                          Control,
                                          Info);
        
	if (status != UMFPACK_OK) {
                umfpack_dl_report_info(Control, Info) ;
                umfpack_dl_report_status(Control, status);
                cleanup();

                throw std::runtime_error("symbolic factorization with umfpack_dl_symbolic failed");
	}

        status = umfpack_dl_numeric(cidx,
                                    ridx,
                                    data,
                                    Symbolic,
                                    &Numeric,
                                    Control,
                                    Info);

        if (status != UMFPACK_OK) {
                umfpack_dl_report_info(Control, Info);
                umfpack_dl_report_status(Control, status);

                cleanup();

                throw std::runtime_error("numeric factorization with umfpack_dl_numeric failed");
        }
}

UmfpackObject::~UmfpackObject()
{
        cleanup();
}

Matrix UmfpackObject::solve(const Matrix& b)
{
        Matrix x(b.rows(), b.columns());

        const octave_idx_type n = A.columns();
        const octave_idx_type* const cidx = A.cidx();
        const octave_idx_type* const ridx = A.ridx();
        const double* const data = A.data();

        double* const xp = x.fortran_vec();
        const double* const bp = b.fortran_vec();

        for (octave_idx_type j = 0; j < b.columns(); ++j) {
                auto status = umfpack_dl_solve(UMFPACK_A,
                                               cidx,
                                               ridx,
                                               data,
                                               xp + j * n,
                                               bp + j * n,
                                               Numeric,
                                               Control,
                                               Info);

                if (status != UMFPACK_OK) {
                        umfpack_dl_report_info(Control, Info);
                        umfpack_dl_report_status(Control, status);

                        throw std::runtime_error("solution with umfpack_dl_solve failed");
                }

		OCTAVE_QUIT;
        }
        
        return x;
}

void UmfpackObject::cleanup()
{
        if (Symbolic) {
                umfpack_dl_free_symbolic(&Symbolic);
        }

        if (Numeric) {
                umfpack_dl_free_numeric(&Numeric);
        }
}

// PKG_ADD: autoload ("umfpack", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("umfpack", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (umfpack, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{umfpack_obj} = umfpack(@var{A}, @var{opt})\n\n"
           "@var{x} = umfpack(@var{umfpack_obj}, @var{b})\n\n"
           "@var{x} = umfpack(@var{A}, @var{b}, @var{opt})\n\n"
           "Solve a system of linear equations by means of Umfpack (http://www.suitesparse.com)\n\n"
           "The first form creates an factor object @var{umfpack_obj} from matrix @var{A}\n\n"
           "After the factor object has been created, the second form uses the factor object @var{umfpack_obj} to solve a system of linear equations @var{A} * @var{x} = @var{b}\n\n"
           "The third form solves a system of linear equations @var{A} * @var{x} = @var{b} but no factor object is returned\n\n"
           "Several options are supported in struct @var{opt}:\n\n"
           "verbose = 0:2\n\n"
           "refine_max_iter @dots{} maximum number of iterations for refinement of the solution\n\n"
           "@end deftypefn\n")
{
        octave_value_list retval;

        if (args.length() < 1 || nargout > 1) {
                print_usage();
                return retval;
        }

        octave_idx_type iarg = 0;
        SparseMatrix A;
        UmfpackObject* pUmfpack = nullptr;
        bool bOwnUmfpack = false;
        bool bHaveMatrix = false;
        bool bHaveRightHandSide = false;
        octave_idx_type n = 0;

        if (args(iarg).is_matrix_type()) {
                A = args(iarg++).sparse_matrix_value();

                if (error_state) {
                        return retval;
                }

                if (A.rows() != A.columns()) {
                        error_with_id("umfpack:input", "pastix: matrix A must be square");
                        return retval;
                }

                if (A.columns() < 1) {
                        error_with_id("umfpack:input", "umfpack: matrix A must have at least one column");
                        return retval;
                }

                n = A.rows();

                bHaveMatrix = true;
        } else {
                octave_base_value& oOctaveObj = const_cast<octave_base_value&>(args(iarg++).get_rep());
                pUmfpack = dynamic_cast<UmfpackObject*>(&oOctaveObj);

                if (!pUmfpack) {
                        error_with_id("umfpack:input", "umfpack: class(umfpack_obj) must be equal to \"umfpack\"");
                        return retval;
                }

                n = pUmfpack->rows();
        }

        Matrix b;

        if (!bHaveMatrix || args.length() > 2) {
                if (args.length() <= iarg) {
                        print_usage();
                        return retval;
                }

                b = args(iarg++).matrix_value();

                if (error_state) {
                        return retval;
                }

                if (b.rows() != n) {
                        error_with_id("umfpack:input", "umfpack: number of rows of b does not match number of rows of A");
                        return retval;
                }

                if (b.columns() < 1) {
                        error_with_id("umfpack:input", "number of columns of b must be at least one");
                        return retval;
                }

                bHaveRightHandSide = true;
        }

        UmfpackObject::Options options;

        if (bHaveMatrix) {
                if (args.length() <= iarg) {
                        print_usage();
                        return retval;
                }

                octave_scalar_map ov_options = args(iarg++).scalar_map_value();

                if (error_state) {
                        return retval;
                }

                auto iter_verbose = ov_options.seek("verbose");

                if (iter_verbose != ov_options.end()) {
                        int verbose = ov_options.contents(iter_verbose).int_value();

                        if (error_state) {
                                return retval;
                        }

                        options.verbose = verbose;
                }

                auto iter_refine_max_iter = ov_options.seek("refine_max_iter");

                if (iter_refine_max_iter != ov_options.end()) {
                        options.refine_max_iter = ov_options.contents(iter_refine_max_iter).int_value();

                        if (error_state) {
                                return retval;
                        }
                }
        }

        try {
                if (bHaveMatrix) {
                        pUmfpack = new UmfpackObject{A, options};

                        bOwnUmfpack = true;
                }

                if (bHaveRightHandSide) {
                        retval.append(pUmfpack->solve(b));

                        if (bOwnUmfpack) {
                                delete pUmfpack;
                                pUmfpack = nullptr;
                        }
                } else {
                        retval.append(pUmfpack);
                }
        } catch (const std::exception& err) {
                if (bOwnUmfpack) {
                        delete pUmfpack;
                        pUmfpack = nullptr;
                }

                error(err.what());
        }

        return retval;
}

/*
%!error umfpack([]);
%!error umfpack(eye(3));
%!error umfpack(eye(3), zeros(3,1));
%!error umfpack(umfpack(eye(3), struct()));
%!error umfpack(eye(3),zeros(2,1),struct());
%!error umfpack(ones(3,2),zeros(3,1),struct());
%!test
%! state = rand("state");
%! unwind_protect
%! rand("seed", 0);
%! n = [2,4,8,16,32,64,128];
%! for e=[0,100]
%!   for u=1:2
%!     for i=1:10
%!       for j=1:numel(n)
%!         for k=1:2
%!           for l=1:2
%!             switch (l)
%!               case 1
%!                 A = rand(n(j),n(j));
%!               otherwise
%!                 A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!             endswitch
%!             Af = A;
%!             b = rand(n(j), 3);
%!             xref = A \ b;
%!             opt.verbose = 2;
%!             opt.refine_max_iter = e;
%!             switch (k)
%!               case 1
%!                 x = umfpack(A, b, opt);
%!               otherwise
%!                 x = umfpack(umfpack(A, opt), b);
%!             endswitch
%!             assert(A * x, b, sqrt(eps) * norm(b));
%!             assert(norm(A * x - b) < sqrt(eps) * norm(A*x+b));
%!             assert(x, xref, sqrt(eps) * norm(x));
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endfor
%! unwind_protect_cleanup
%! rand("state", state);
%! end_unwind_protect
*/
