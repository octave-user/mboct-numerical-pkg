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

template <typename T>
struct UmfpackTraits;

class UmfpackObjectDouble;

class UmfpackObjectComplex;

template <>
struct UmfpackTraits<double> {
     typedef UmfpackObjectDouble UmfpackObjectType;
     typedef SparseMatrix SparseMatrixType;
     typedef Matrix DenseMatrixType;

     UmfpackTraits()=default;

     explicit UmfpackTraits(const SparseMatrixType& A)
	  :A(A) {
     }

     dim_vector dims() const { return A.dims(); }
     octave_idx_type columns() const { return A.columns(); }
     
     SuiteSparse_long umfpack_symbolic(void **Symbolic, const double Control[], double Info[]) const {
	  return umfpack_dl_symbolic(A.rows(), A.cols(), A.cidx(), A.ridx(), A.data(), Symbolic, Control, Info);
     }

     SuiteSparse_long umfpack_numeric(void *Symbolic, void **Numeric, const double Control[], double Info[]) const {
	  return umfpack_dl_numeric(A.cidx(), A.ridx(), A.data(), Symbolic, Numeric, Control, Info);
     }

     SuiteSparse_long umfpack_solve(SuiteSparse_long sys, double X[], const double B[], void *Numeric, const double Control[], double Info[]) const {
	  return umfpack_dl_solve(sys, A.cidx(), A.ridx(), A.data(), X, B, Numeric, Control, Info);
     }

     static constexpr auto umfpack_defaults = umfpack_dl_defaults;
     static constexpr auto umfpack_report_info = umfpack_dl_report_info;
     static constexpr auto umfpack_report_status = umfpack_dl_report_status;
     static constexpr auto umfpack_free_symbolic = umfpack_dl_free_symbolic;
     static constexpr auto umfpack_free_numeric = umfpack_dl_free_numeric;
     static constexpr auto sparse_matrix_value = &octave_value::sparse_matrix_value;
     static constexpr auto matrix_value = &octave_value::matrix_value;
     static constexpr bool isreal = true;
     static constexpr bool iscomplex = false;
     
private:
     const SparseMatrixType A;
};

template <>
struct UmfpackTraits<std::complex<double> > {
     typedef UmfpackObjectComplex UmfpackObjectType;
     typedef SparseComplexMatrix SparseMatrixType;
     typedef ComplexMatrix DenseMatrixType;

     UmfpackTraits()=default;

     explicit UmfpackTraits(const SparseMatrixType& A)
	  :A(A),
	   Are(A.nnz()),
	   Aim(A.nnz()),
	   Xre(A.cols()),
	   Xim(A.cols()),
	   Bre(A.rows()),
	   Bim(A.rows()) {

	  const std::complex<double>* const data = A.data();
	  
	  for (octave_idx_type i = 0; i < A.nnz(); ++i) {
	       Are.xelem(i) = std::real(data[i]);
	       Aim.xelem(i) = std::imag(data[i]);
	  }
     }

     dim_vector dims() const { return A.dims(); }
     octave_idx_type columns() const { return A.columns(); }
     
     SuiteSparse_long umfpack_symbolic(void **Symbolic, const double Control [], double Info []) const {
	  return umfpack_zl_symbolic(A.rows(), A.cols(), A.cidx(), A.ridx(), Are.fortran_vec(), Aim.fortran_vec(), Symbolic, Control, Info);
     }

     SuiteSparse_long umfpack_numeric(void *Symbolic, void **Numeric, const double Control[], double Info[]) const {
	  return umfpack_zl_numeric(A.cidx(), A.ridx(), Are.fortran_vec(), Aim.fortran_vec(), Symbolic, Numeric, Control, Info);
     }

     SuiteSparse_long umfpack_solve(SuiteSparse_long sys, std::complex<double> X[], const std::complex<double> B[], void *Numeric, const double Control[], double Info[]) {
	  for (octave_idx_type i = 0; i < A.rows(); ++i) {
	       Bre.xelem(i) = std::real(B[i]);
	       Bim.xelem(i) = std::imag(B[i]);
	  }
	  
	  auto status = umfpack_zl_solve(sys, A.cidx(), A.ridx(), Are.fortran_vec(), Aim.fortran_vec(), Xre.fortran_vec(), Xim.fortran_vec(), Bre.fortran_vec(), Bim.fortran_vec(), Numeric, Control, Info);

	  for (octave_idx_type i = 0; i < A.cols(); ++i) {
	       X[i] = std::complex<double>(Xre.xelem(i), Xim.xelem(i));
	  }
	  
	  return status;
     }
     
     static constexpr auto umfpack_defaults = umfpack_zl_defaults;
     static constexpr auto umfpack_report_info = umfpack_zl_report_info;
     static constexpr auto umfpack_report_status = umfpack_zl_report_status;
     static constexpr auto umfpack_free_symbolic = umfpack_zl_free_symbolic;
     static constexpr auto umfpack_free_numeric = umfpack_zl_free_numeric;
     static constexpr auto sparse_matrix_value = &octave_value::sparse_complex_matrix_value;
     static constexpr auto matrix_value = &octave_value::complex_matrix_value;
     static constexpr bool isreal = false;
     static constexpr bool iscomplex = true;

private:
     const SparseMatrixType A;
     ColumnVector Are, Aim, Xre, Xim, Bre, Bim;
};

template <typename T>
class UmfpackObject : public octave_base_value, private UmfpackTraits<T> {
     typedef typename UmfpackTraits<T>::SparseMatrixType SparseMatrixType;
     typedef typename UmfpackTraits<T>::DenseMatrixType DenseMatrixType;
     typedef typename UmfpackTraits<T>::UmfpackObjectType UmfpackObjectType;
     
public:
     struct Options {
	  int verbose = UMFPACK_DEFAULT_PRL;
	  int refine_max_iter = UMFPACK_DEFAULT_IRSTEP;
     };

     UmfpackObject();
     explicit UmfpackObject(const SparseMatrixType& A, const Options& options);
     virtual ~UmfpackObject(void);
     DenseMatrixType solve(const DenseMatrixType& b);
     virtual bool is_constant(void) const{ return true; }
     virtual bool is_defined(void) const{ return true; }
     virtual dim_vector dims (void) const { return UmfpackTraits<T>::dims(); }
     virtual void print(std::ostream& os, bool pr_as_read_syntax) {
     }
     virtual bool isreal() const { return UmfpackTraits<T>::isreal; }
     virtual bool iscomplex() const { return UmfpackTraits<T>::iscomplex; }
     static octave_value_list eval(const octave_value_list& args, int nargout);
     
private:
     void cleanup();

     Options options;
     double Control[UMFPACK_CONTROL];
     double Info[UMFPACK_INFO];
     void *Symbolic = nullptr;
     void *Numeric = nullptr;
};

class UmfpackObjectDouble: public UmfpackObject<double> {
public:
     UmfpackObjectDouble()=default;

     template <typename... Args>
     UmfpackObjectDouble(const Args&... args): UmfpackObject<double>(args...) {
     }
private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

class UmfpackObjectComplex: public UmfpackObject<std::complex<double> > {
public:
     UmfpackObjectComplex()=default;
     template <typename... Args>
     UmfpackObjectComplex(const Args&... args): UmfpackObject<std::complex<double> >(args...) {
     }
private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(UmfpackObjectDouble, "umfpackd", "umfpackd")
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(UmfpackObjectComplex, "umfpackc", "umfpackc")

template <typename T>
UmfpackObject<T>::UmfpackObject()
{
}

template <typename T>
UmfpackObject<T>::UmfpackObject(const SparseMatrixType& A, const Options& opt)
     : UmfpackTraits<T>(A),
       options(opt)
{

     if (A.rows() != A.columns()) {
	  throw std::runtime_error("matrix A must be square");
     }

     std::memset(&Info[0], 0, sizeof(Info));

     this->umfpack_defaults(Control);

     Control[UMFPACK_PRL] = opt.verbose;
     Control[UMFPACK_IRSTEP] = opt.refine_max_iter;

     auto status = this->umfpack_symbolic(&Symbolic, Control, Info);

     if (status != UMFPACK_OK) {
	  this->umfpack_report_info(Control, Info) ;
	  this->umfpack_report_status(Control, status);
	  cleanup();

	  throw std::runtime_error("symbolic factorization with umfpack_dl_symbolic failed");
     }

     status = this->umfpack_numeric(Symbolic, &Numeric, Control, Info);

     if (status != UMFPACK_OK) {
	  this->umfpack_report_info(Control, Info);
	  this->umfpack_report_status(Control, status);

	  cleanup();

	  throw std::runtime_error("numeric factorization with umfpack_dl_numeric failed");
     }
}

template <typename T>
UmfpackObject<T>::~UmfpackObject()
{
     cleanup();
}

template <typename T>
typename UmfpackObject<T>::DenseMatrixType UmfpackObject<T>::solve(const DenseMatrixType& b)
{
     DenseMatrixType x(b.rows(), b.columns());

     const octave_idx_type n = this->columns();

     T* const xp = x.fortran_vec();
     const T* const bp = b.fortran_vec();

     for (octave_idx_type j = 0; j < b.columns(); ++j) {
	  auto status = this->umfpack_solve(UMFPACK_A,
					    xp + j * n,
					    bp + j * n,
					    Numeric,
					    Control,
					    Info);

	  if (status != UMFPACK_OK) {
	       this->umfpack_report_info(Control, Info);
	       this->umfpack_report_status(Control, status);

	       throw std::runtime_error("solution with umfpack_dl_solve failed");
	  }

	  OCTAVE_QUIT;
     }

     return x;
}

template <typename T>
void UmfpackObject<T>::cleanup()
{
     if (Symbolic) {
	  this->umfpack_free_symbolic(&Symbolic);
     }

     if (Numeric) {
	  this->umfpack_free_numeric(&Numeric);
     }
}

template <typename T>
octave_value_list UmfpackObject<T>::eval(const octave_value_list& args, int nargout)
{
     octave_value_list retval;

     octave_idx_type iarg = 0;
     SparseMatrixType A;
     UmfpackObject<T>* pUmfpack = nullptr;
     bool bOwnUmfpack = false;
     bool bHaveMatrix = false;
     bool bHaveRightHandSide = false;
     octave_idx_type n = 0;

     if (args(iarg).is_matrix_type()) {
	  A = (args(iarg++).*UmfpackTraits<T>::sparse_matrix_value)(false);

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
	  pUmfpack = dynamic_cast<UmfpackObject<T>*>(&oOctaveObj);

	  if (!pUmfpack) {
	       error_with_id("umfpack:input", "umfpack: class(umfpack_obj) must be equal to \"umfpack\"");
	       return retval;
	  }

	  n = pUmfpack->rows();
     }

     DenseMatrixType b;

     if (!bHaveMatrix || args.length() > 2) {
	  if (args.length() <= iarg) {
	       print_usage();
	       return retval;
	  }

	  if (args(1).iscomplex() && args(0).isreal()) {
	       error_with_id("umfpack:input", "umfpack: complex right hand side cannot be used with real matrix");
	       return retval;
	  }

	  b = (args(iarg++).*UmfpackTraits<T>::matrix_value)(false);

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

     Options options;

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
	       pUmfpack = new UmfpackObjectType{A, options};

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

	  error_with_id("umfpack:exception", "%s", err.what());
     }

     return retval;
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

     bool bcomplex = args(0).iscomplex();

     if (args.length() > 1 && args(1).is_matrix_type()) {
	  bcomplex = bcomplex || args(1).iscomplex();
     }
     
     if (bcomplex) {
	  retval = UmfpackObject<std::complex<double> >::eval(args, nargout);
     } else {
	  retval = UmfpackObject<double>::eval(args, nargout);
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
