// Copyright (C) 2021(-2023) Reinhard <octave-user@a1.net>

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
#include <algorithm>
#include <memory>
#include <cstdint>
#include <octave/oct.h>

#include <StrumpackSparseSolver.hpp>

class StrumpackObjectDouble;
class StrumpackObjectComplex;

template <typename T>
struct StrumpackTraits;

template <>
struct StrumpackTraits<double> {
     typedef SparseMatrix SparseMatrixType;
     typedef Matrix DenseMatrixType;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_matrix_value;
     static constexpr auto matrix_value = &octave_value::matrix_value;

     static bool isfinite(double x) {
          return std::isfinite(x);
     }

     typedef StrumpackObjectDouble StrumpackObjectType;
     static constexpr bool isreal = true;
     static constexpr bool iscomplex = false;
};

template <>
struct StrumpackTraits<std::complex<double> > {
     typedef SparseComplexMatrix SparseMatrixType;
     typedef ComplexMatrix DenseMatrixType;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_complex_matrix_value;
     static constexpr auto matrix_value = &octave_value::complex_matrix_value;

     static bool isfinite(const std::complex<double>& z) {
          return std::isfinite(std::real(z)) && std::isfinite(std::imag(z));
     }

     typedef StrumpackObjectComplex StrumpackObjectType;
     static constexpr bool isreal = false;
     static constexpr bool iscomplex = true;
};

template <typename T>
class StrumpackObject : public octave_base_value {
     typedef typename StrumpackTraits<T>::SparseMatrixType SparseMatrixType;
     typedef typename StrumpackTraits<T>::DenseMatrixType DenseMatrixType;
     typedef typename StrumpackTraits<T>::StrumpackObjectType StrumpackObjectType;
public:
     struct Options {
          bool verbose = false;
          bool symmetric = false;
          strumpack::CompressionType compression = strumpack::CompressionType::NONE;
          double compression_rel_tol = 1e-2;
          double compression_abs_tol = 1e-8;
          strumpack::ReorderingStrategy ordering = strumpack::ReorderingStrategy::SCOTCH;
          double relative_tol = 1e-6;
          double absolute_tol = 1e-10;
          int restart = 30;
          int refine_max_iter = 5000;
     };

     StrumpackObject();
     explicit StrumpackObject(const SparseMatrixType& A, const Options& options);
     virtual ~StrumpackObject(void);
     virtual size_t byte_size() const;
     virtual dim_vector dims() const;
     bool solve(DenseMatrixType& b, DenseMatrixType& x) const;
     static bool get_options(const octave_value& ovOptions, StrumpackObject::Options& options);
     virtual bool is_constant(void) const{ return true; }
     virtual bool is_defined(void) const{ return true; }
     virtual bool isreal() const { return StrumpackTraits<T>::isreal; }
     virtual bool iscomplex() const { return StrumpackTraits<T>::iscomplex; }
     static octave_value_list eval(const octave_value_list& args, int nargout);

private:
     void cleanup();

     Options options;
     mutable strumpack::StrumpackSparseSolver<T, octave_idx_type> oSolver;
     const octave_idx_type ncols;
};

class StrumpackObjectDouble: public StrumpackObject<double> {
public:
     StrumpackObjectDouble()=default;

     template <typename... Args>
     StrumpackObjectDouble(const Args&... args)
          :StrumpackObject<double>(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

class StrumpackObjectComplex: public StrumpackObject<std::complex<double> > {
public:
     StrumpackObjectComplex()=default;

     template <typename... Args>
     StrumpackObjectComplex(const Args&... args)
          :StrumpackObject<std::complex<double> >(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(StrumpackObjectDouble, "strumpackd", "strumpackd")
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(StrumpackObjectComplex, "strumpackc", "strumpackc")

template <typename T>
StrumpackObject<T>::StrumpackObject()
:ncols(0)
{
}

template <typename T>
StrumpackObject<T>::StrumpackObject(const SparseMatrixType& A, const Options& options)
     :options(options), ncols(A.columns())
{
     using namespace strumpack;
     const SparseMatrixType Atmp = options.symmetric ? A : A.transpose();
     const T* values = Atmp.data();
     const octave_idx_type* rowptr = Atmp.cidx();
     const octave_idx_type* colptr = Atmp.ridx();

     oSolver.set_csr_matrix(Atmp.columns(), rowptr, colptr, values, options.symmetric);

     oSolver.options().set_maxit(std::max(1, options.refine_max_iter));
     oSolver.options().set_verbose(options.verbose);
     oSolver.options().set_compression(options.compression);
     oSolver.options().set_compression_rel_tol(std::max(0., options.compression_rel_tol));
     oSolver.options().set_compression_abs_tol(std::max(0., options.compression_abs_tol));
     oSolver.options().set_reordering_method(options.ordering);
     oSolver.options().set_rel_tol(std::max(0., options.relative_tol));
     oSolver.options().set_abs_tol(std::max(0., options.absolute_tol));
     oSolver.options().set_gmres_restart(std::max(1, options.restart));

     ReturnCode rc = oSolver.reorder();

     if (ReturnCode::SUCCESS != rc) {
          throw std::runtime_error("failed to compute ordering");
     }

     rc = oSolver.factor();

     if (ReturnCode::SUCCESS != rc) {
          throw std::runtime_error("failed to factor matrix");
     }
}

template <typename T>
size_t StrumpackObject<T>::byte_size() const
{
     return oSolver.factor_memory();
}

template <typename T>
dim_vector StrumpackObject<T>::dims() const
{
     return dim_vector(ncols, ncols);
}

template <typename T>
StrumpackObject<T>::~StrumpackObject()
{
}

template <typename T>
bool StrumpackObject<T>::solve(DenseMatrixType& b, DenseMatrixType& x) const {
     using namespace strumpack;

     if (b.rows() != ncols) {
          error_with_id("strumpack:solve", "strumpack: rows(b)=%ld must be equal to rows(A)=%ld", long(b.rows()), long(ncols));
          return false;
     }

     T* px = x.fortran_vec();
     const T* pb = b.fortran_vec();

     for (octave_idx_type i = 0; i < b.columns(); ++i) {
          ReturnCode rc = oSolver.solve(pb, px);

          OCTAVE_QUIT;

          if (ReturnCode::SUCCESS != rc) {
               error_with_id("strumpack:solve", "strumpack solve failed with status %d", static_cast<int>(rc));
               return false;
          }

          px += x.rows();
          pb += b.rows();
     }

     return true;
}

template <typename T>
bool StrumpackObject<T>::get_options(const octave_value& ovOptions, StrumpackObject::Options& options)
{
     using namespace strumpack;

     const octave_scalar_map om_options = ovOptions.scalar_map_value();

     {
          const auto imat_type = om_options.seek("symmetric");

          if (imat_type != om_options.end()) {
               options.symmetric = om_options.contents(imat_type).bool_value();
          }
     }

     {
          const auto iverbose = om_options.seek("verbose");

          if (iverbose != om_options.end()) {
               options.verbose = om_options.contents(iverbose).bool_value();
          }
     }

     {
          const auto irefine = om_options.seek("refine_max_iter");

          if (irefine != om_options.end()) {
               octave_value ov_ref = om_options.contents(irefine);

               options.refine_max_iter = ov_ref.int_value();
          }
     }

     {
          const auto iter_compress = om_options.seek("compression");

          if (iter_compress != om_options.end()) {
               options.compression = static_cast<CompressionType>(om_options.contents(iter_compress).int_value());
          }
     }

     {
          const auto iter_compress_rel_tol = om_options.seek("compression_rel_tol");

          if (iter_compress_rel_tol != om_options.end()) {
               options.compression_rel_tol = om_options.contents(iter_compress_rel_tol).scalar_value();
          }
     }

     {
          const auto iter_compress_abs_tol = om_options.seek("compression_abs_tol");

          if (iter_compress_abs_tol != om_options.end()) {
               options.compression_abs_tol = om_options.contents(iter_compress_abs_tol).scalar_value();
          }
     }

     {
          const auto iter_ordering = om_options.seek("ordering");

          if (iter_ordering != om_options.end()) {
               options.ordering = static_cast<ReorderingStrategy>(om_options.contents(iter_ordering).int_value());
          }
     }

     {
          const auto iter_abs_tol = om_options.seek("absolute_tol");

          if (iter_abs_tol != om_options.end()) {
               options.absolute_tol = om_options.contents(iter_abs_tol).scalar_value();
          }
     }

     {
          const auto iter_rel_tol = om_options.seek("relative_tol");

          if (iter_rel_tol != om_options.end()) {
               options.relative_tol = om_options.contents(iter_rel_tol).scalar_value();
          }
     }

     {
          const auto iter_restart = om_options.seek("restart");

          if (iter_restart != om_options.end()) {
               options.restart = om_options.contents(iter_restart).int_value();
          }
     }

     return true;
}

template<typename T>
octave_value_list StrumpackObject<T>::eval(const octave_value_list& args, int nargout)
{
     octave_value_list retval;
     octave_idx_type iarg = 0;
     SparseMatrixType A;
     StrumpackObject<T>* pStrumpack = nullptr;
     bool bOwnStrumpack = false;
     bool bHaveMatrix = false;
     bool bHaveRightHandSide = false;

     if (args(iarg).is_matrix_type()) {
          A = (args(iarg++).*StrumpackTraits<T>::sparse_matrix_value)(false);

          if (A.rows() != A.columns()) {
               error_with_id("strumpack:input", "strumpack: matrix A must be square");
               return retval;
          }

          if (A.columns() < 1) {
               error_with_id("strumpack:input", "strumpack: matrix A must have at least one column");
               return retval;
          }

          bHaveMatrix = true;
     } else {
          octave_base_value& oOctaveObj = const_cast<octave_base_value&>(args(iarg++).get_rep());
          pStrumpack = dynamic_cast<StrumpackObject<T>*>(&oOctaveObj);

          if (!pStrumpack) {
               error_with_id("strumpack:input", "strumpack: class(strumpack_obj) must be equal to \"strumpack\"");
               return retval;
          }
     }

     DenseMatrixType x, b;

     if (!bHaveMatrix || args.length() >= 3) {
          if (args(0).isreal() && !args(0).is_matrix_type() && args(1).iscomplex()) {
               error_with_id("strumpack:input", "strumpack: complex right hand side cannot be used with a real matrix");
               return retval;
          }

          b = (args(iarg++).*StrumpackTraits<T>::matrix_value)(false);
          x.resize(b.rows(), b.columns());

          bHaveRightHandSide = true;
     }

     if (bHaveMatrix) {
          if (args.length() <= iarg) {
               error_with_id("strumpack:input", "strumpack: missing argument \"options\"");
               return retval;
          }

          Options options;

          if (!get_options(args(iarg++), options)) {
               return retval;
          }

          try {
               pStrumpack = new StrumpackObjectType{A, options};
          } catch(const std::exception& err) {
               error_with_id("strumpack:factor", "%s", err.what());
               return retval;
          }

          bOwnStrumpack = true;
     }

     if (bHaveRightHandSide) {
          if (pStrumpack->solve(b, x)) {
               retval.append(x);
          }

          if (bOwnStrumpack) {
               delete pStrumpack;
               pStrumpack = nullptr;
          }
     } else {
          retval.append(pStrumpack);
     }

     return retval;
}

// PKG_ADD: autoload ("strumpack", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("strumpack", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (strumpack, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{strumpack_obj} = strumpack(@var{A}, @var{options})\n\n"
           "@var{x} = strumpack(@var{strumpack_obj}, @var{b})\n\n"
           "@var{x} = strumpack(@var{A}, @var{b}, @var{options})\n\n"
           "Solve a system of linear equations by means of Strumpack (https://github.com/pghysels/STRUMPACK)\n\n"
           "The first form creates an factor object @var{strumpack_obj} from matrix @var{A}\n\n"
           "After the factor object has been created, the second form uses the factor object @var{strumpack_obj} to solve a system of linear equations @var{A} * @var{x} = @var{b}\n\n"
           "The third form solves a system of linear equations @var{A} * @var{x} = @var{b} but no factor object is returned\n\n"
           "Several options are supported in struct @var{options}:\n\n"
           "@var{options}.verbose = {true|false}\n\n"
           "@var{options}.symmetric = {true|false}\n\n"
           "@var{options}.compression = {STRUMPACK_COMPRESS_NONE|STRUMPACK_COMPRESS_HSS|STRUMPACK_COMPRESS_BLR|STRUMPACK_COMPRESS_HODLR|STRUMPACK_COMPRESS_LOSSLESS|STRUMPACK_COMPRESS_LOSSLY}\n\n"
           "@var{options}.ordering = {STRUMPACK_ORDERING_NATURAL|STRUMPACK_ORDERING_METIS|STRUMPACK_ORDERING_PARMETIS|STRUMPACK_ORDERING_SCOTCH|STRUMPACK_ORDERING_PTSCOTCH|STRUMPACK_ORDERING_RCM|STRUMPACK_ORDERING_GEOMETRIC}\n\n"
           "@var{options}.compression_rel_tol @dots{} relative tolerance for the compressed representation of the matrix\n\n"
           "@var{options}.compression_abs_tol @dots{} absolute tolerance for the compressed representation of the matrix\n\n"
           "@var{options}.absolute_tol @dots{} absolute tolerance for the solution\n\n"
           "@var{options}.refine_max_iter @dots{} maximum number of iterations for refinement of the solution\n\n"
           "@var{options}.relative_tol @dots{} relative tolerance for the solution\n\n"
           "@var{options}.restart @dots{} number of restarts for Gmres method\n\n"
           "@end deftypefn\n")
{
     octave_value_list retval;

     if (args.length() < 2 || nargout > 1) {
          print_usage();
          return retval;
     }

     bool bcomplex = args(0).iscomplex();

     if (args.length() > 1 && args(1).is_matrix_type()) {
          bcomplex = bcomplex || args(1).iscomplex();
     }

     if (bcomplex) {
          retval = StrumpackObject<std::complex<double> >::eval(args, nargout);
     } else {
          retval = StrumpackObject<double>::eval(args, nargout);
     }

     return retval;
}

#define DEFINE_GLOBAL_CONSTANT(CONST,VALUE)                             \
     DEFUN_DLD(STRUMPACK_##CONST, args, nargout, "id = STRUMPACK_" #CONST  "()\n") \
     {                                                                  \
          return octave_value(octave_int32(VALUE));                     \
     }

DEFINE_GLOBAL_CONSTANT(COMPRESS_NONE, strumpack::CompressionType::NONE)
DEFINE_GLOBAL_CONSTANT(COMPRESS_HSS, strumpack::CompressionType::HSS)
DEFINE_GLOBAL_CONSTANT(COMPRESS_BLR, strumpack::CompressionType::BLR)
DEFINE_GLOBAL_CONSTANT(COMPRESS_HODLR, strumpack::CompressionType::HODLR)
DEFINE_GLOBAL_CONSTANT(COMPRESS_LOSSLESS, strumpack::CompressionType::LOSSLESS)
DEFINE_GLOBAL_CONSTANT(COMPRESS_LOSSY, strumpack::CompressionType::LOSSY)
DEFINE_GLOBAL_CONSTANT(ORDERING_NATURAL, strumpack::ReorderingStrategy::NATURAL)
DEFINE_GLOBAL_CONSTANT(ORDERING_SCOTCH, strumpack::ReorderingStrategy::SCOTCH)
DEFINE_GLOBAL_CONSTANT(ORDERING_PTSCOTCH, strumpack::ReorderingStrategy::PTSCOTCH)
DEFINE_GLOBAL_CONSTANT(ORDERING_METIS, strumpack::ReorderingStrategy::METIS)
DEFINE_GLOBAL_CONSTANT(ORDERING_PARMETIS, strumpack::ReorderingStrategy::PARMETIS)
DEFINE_GLOBAL_CONSTANT(ORDERING_RCM, strumpack::ReorderingStrategy::RCM)
DEFINE_GLOBAL_CONSTANT(ORDERING_GEOMETRIC, strumpack::ReorderingStrategy::GEOMETRIC)

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_NONE", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_NONE", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_HSS", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_HSS", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_BLR", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_BLR", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_HODLR", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_HODLR", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_LOSSLESS", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_LOSSLESS", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_COMPRESS_LOSSY", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_COMPRESS_LOSSY", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_NATURAL", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_NATURAL", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_METIS", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_METIS", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_PARMETIS", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_PARMETIS", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_SCOTCH", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_SCOTCH", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_PTSCOTCH", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_PTSCOTCH", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_RCM", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_RCM", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("STRUMPACK_ORDERING_GEOMETRIC", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("STRUMPACK_ORDERING_GEOMETRIC", "__mboct_numerical__.oct", "remove");
