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

#define NDEBUG

#include <stdexcept>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#include <iostream>
#endif
#include <vector>
#include <octave/oct.h>

#include MKL_PARDISO_H
#include MKL_SERVICE_H

class PardisoObjectDouble;
class PardisoObjectComplex;

template <typename T>
struct PardisoTraits;

template <>
struct PardisoTraits<double> {
     typedef SparseMatrix SparseMatrixType;
     typedef Matrix DenseMatrixType;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_matrix_value;
     static constexpr auto matrix_value = &octave_value::matrix_value;

     static bool isfinite(double x) {
          return std::isfinite(x);
     }

     typedef PardisoObjectDouble PardisoObjectType;
     static constexpr bool isreal = true;
     static constexpr bool iscomplex = false;
     static constexpr long long mtypesym = -2;
     static constexpr long long mtypeunsym = 11;

#ifndef NDEBUG
     static constexpr double nan = NAN;
#endif
};

#ifndef NDEBUG
constexpr double PardisoTraits<double>::nan;
#endif

template <>
struct PardisoTraits<std::complex<double> > {
     typedef SparseComplexMatrix SparseMatrixType;
     typedef ComplexMatrix DenseMatrixType;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_complex_matrix_value;
     static constexpr auto matrix_value = &octave_value::complex_matrix_value;

     static bool isfinite(const std::complex<double>& z) {
          return std::isfinite(std::real(z)) && std::isfinite(std::imag(z));
     }

     typedef PardisoObjectComplex PardisoObjectType;
     static constexpr bool isreal = false;
     static constexpr bool iscomplex = true;
     static constexpr long long mtypesym = 6;
     static constexpr long long mtypeunsym = 13;

#ifndef NDEBUG
     static constexpr std::complex<double> nan = {NAN, NAN};
#endif
};

#ifndef NDEBUG
constexpr std::complex<double> PardisoTraits<std::complex<double> >::nan;
#endif

template <typename T>
class PardisoObject : public octave_base_value {
     typedef typename PardisoTraits<T>::SparseMatrixType SparseMatrixType;
     typedef typename PardisoTraits<T>::DenseMatrixType DenseMatrixType;
     typedef typename PardisoTraits<T>::PardisoObjectType PardisoObjectType;
public:
     struct Options {
          bool verbose = false;
          bool symmetric = false;
          int refine_max_iter = 3;
          int number_of_threads = 1;
#ifdef PARDISO_USE_OUT_OF_CORE_MODE
          int out_of_core_mode = 0;
#endif
          int ordering = 1;
          int scaling = 0;
          int weighted_matching = 0;
     };

     PardisoObject();
     explicit PardisoObject(const SparseMatrixType& A, const Options& options);
     virtual ~PardisoObject(void);
     virtual size_t byte_size() const;
     virtual dim_vector dims() const;
     bool solve(DenseMatrixType& b, DenseMatrixType& x, long long sys) const;
     static bool get_options(const octave_value& ovOptions, PardisoObject::Options& options);
     virtual bool is_constant(void) const{ return true; }
     virtual bool is_defined(void) const{ return true; }
     virtual bool isreal() const { return PardisoTraits<T>::isreal; }
     virtual bool iscomplex() const { return PardisoTraits<T>::iscomplex; }
     static octave_value_list eval(const octave_value_list& args, int nargout);

private:
     void init();
     void cleanup();
     long long pardiso(T* b, T* x, long long nrhs) const;

     const Options options;
     mutable _MKL_DSS_HANDLE_t pt[64];
     mutable long long iparm[64];
     std::vector<T> Ax;
     std::vector<long long> Ai;
     std::vector<long long> Ap;
     mutable long long phase = 12LL;
     mutable long long maxfct = 1LL;
     mutable long long mnum = 1LL;
     mutable long long mtype = -2LL;
     mutable long long n = 0LL;
     mutable long long msglvl = 0LL;
};

class PardisoObjectDouble: public PardisoObject<double> {
public:
     PardisoObjectDouble()=default;

     template <typename... Args>
     PardisoObjectDouble(const Args&... args)
          :PardisoObject<double>(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

class PardisoObjectComplex: public PardisoObject<std::complex<double> > {
public:
     PardisoObjectComplex()=default;

     template <typename... Args>
     PardisoObjectComplex(const Args&... args)
          :PardisoObject<std::complex<double> >(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(PardisoObjectDouble, "pardisod", "pardisod")
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(PardisoObjectComplex, "pardisoc", "pardisoc")

template <typename T>
PardisoObject<T>::PardisoObject()
{
     init();
}

template <typename T>
PardisoObject<T>::PardisoObject(const SparseMatrixType& A, const Options& options)
     :options(options),
      mtype(options.symmetric ? PardisoTraits<T>::mtypesym : PardisoTraits<T>::mtypeunsym),
      n(A.columns()),
      msglvl(options.verbose)
{
     init();

     const T* const data = A.data();
     const octave_idx_type* const ridx = A.ridx();
     const octave_idx_type* const cidx = A.cidx();

     enum MatType {
          MAT_DIAG,
          MAT_SYM_UPPER,
          MAT_SYM_LOWER,
          MAT_FULL
     } eMatType = MAT_DIAG;

     if (options.symmetric) {
          for (octave_idx_type j = 0L; j < n; ++j) {
               for (octave_idx_type i = cidx[j]; i < cidx[j + 1L]; ++i) {
                    switch (eMatType) {
                    case MAT_DIAG:
                         if (ridx[i] > j) {
                              eMatType = MAT_SYM_LOWER;
                         } else if (ridx[i] < j) {
                              eMatType = MAT_SYM_UPPER;
                         }
                         break;
                    case MAT_SYM_UPPER:
                         if (ridx[i] > j) {
                              eMatType = MAT_FULL;
                              goto exit_mat_type;
                         }
                         break;
                    case MAT_SYM_LOWER:
                         if (ridx[i] < j) {
                              eMatType = MAT_FULL;
                              goto exit_mat_type;
                         }
                         break;
                    case MAT_FULL:
                         goto exit_mat_type;
                    }
               }
          }
     exit_mat_type:
          ;
          if (eMatType == MAT_FULL || eMatType == MAT_DIAG) {
               // The full symmetric matrix has been provided,
               // so we must copy either the upper or lower triangular part.
               eMatType = MAT_SYM_UPPER;
          }
     } else {
          // The full unsymmetric matrix must be provided
          eMatType = MAT_FULL;
     }

     std::vector<long long> nnz(n, 0LL);
     std::vector<bool> diag;

     switch (eMatType) {
     case MAT_SYM_UPPER:
     case MAT_SYM_LOWER:
          diag.resize(n, false);
          break;
     default:
          break;
     }

     const bool transpose = eMatType == MAT_SYM_LOWER;

     for (octave_idx_type j = 0L; j < n; ++j) {
          for (octave_idx_type i = cidx[j]; i < cidx[j + 1L]; ++i) {
               const long long irow = transpose ? j : ridx[i];
               const long long icol = transpose ? ridx[i] : j;

               switch (eMatType) {
               case MAT_SYM_UPPER:
               case MAT_SYM_LOWER:
                    if (irow == icol) {
                         diag[irow] = true;
                    }
                    break;

               default:
                    break;
               }

               if (eMatType == MAT_FULL || irow <= icol) {
                    ++nnz[irow];
               }
          }
     }

     switch (eMatType) {
     case MAT_SYM_UPPER:
     case MAT_SYM_LOWER:
          for (octave_idx_type i = 0L; i < n; ++i) {
               if (!diag[i]) {
                    ++nnz[i];
               }
          }
          break;
     default:
          break;
     }

     long long nnztot = 0LL;

     for (long long nnzi: nnz) {
          nnztot += nnzi;
     }

     Ax.resize(nnztot);
     Ai.resize(nnztot);
     Ap.resize(n + 1LL);

#ifndef NDEBUG
     std::fill(std::begin(Ax), std::end(Ax), PardisoTraits<T>::nan);
     std::fill(std::begin(Ai), std::end(Ai), std::numeric_limits<long long>::min());
     std::fill(std::begin(Ap), std::end(Ap), std::numeric_limits<long long>::min());
#endif

     Ap[0] = 0LL;

     for (long long i = 0LL; i < n; ++i) {
          Ap[i + 1LL] = Ap[i] + nnz[i];
     }

     assert(nnztot == Ap[n]);

     std::fill(std::begin(nnz), std::end(nnz), 0LL);

     switch (eMatType) {
     case MAT_SYM_UPPER:
     case MAT_SYM_LOWER:
          for (octave_idx_type irow = 0L; irow < n; ++irow) {
               if (!diag[irow]) {
                    assert(nnz[irow] == 0LL);
                    assert(nnz[irow] < Ap[irow + 1LL] - Ap[irow]);

                    const long long idx = Ap[irow];

                    assert(idx >= 0);
                    assert(idx < nnztot);

                    Ax[idx] = 0.0;
                    Ai[idx] = irow;

                    ++nnz[irow];

                    assert(nnz[irow] <= Ap[irow + 1LL] - Ap[irow]);
               }
          }
          break;
     default:
          break;
     }

     for (octave_idx_type j = 0L; j < n; ++j) {
          for (octave_idx_type i = cidx[j]; i < cidx[j + 1L]; ++i) {
               const long long irow = transpose ? j : ridx[i];
               const long long icol = transpose ? ridx[i] : j;

               long long& ptr = nnz[irow];

               if (eMatType == MAT_FULL || irow <= icol) {
                    assert(ptr >= 0LL);
                    assert(ptr < Ap[irow + 1LL] - Ap[irow]);

                    const long long idx = Ap[irow] + ptr;

                    assert(idx >= 0);
                    assert(idx < nnztot);

                    Ai[idx] = icol;
                    Ax[idx] = data[i];

                    ++ptr;

                    assert(ptr <= Ap[irow + 1LL] - Ap[irow]);
               }
          }
     }

#ifndef NDEBUG
     for (long long i = 0LL; i < n; ++i) {
          assert(nnz[i] == Ap[i + 1] - Ap[i]);
          for (long long j = Ap[i]; j < Ap[i + 1]; ++j) {
               std::cerr << "A(" << i + 1LL << ", " << Ai[j] + 1LL << ")=" << Ax[j] << "\n";
          }
     }
#endif

     iparm[0] = 1LL; // Use default values.
     iparm[1] = options.ordering;
     iparm[7] = options.refine_max_iter; // Iterative refinement step
     iparm[10] = options.scaling;
     iparm[12] = options.weighted_matching;
     iparm[34] = 1LL; // Zero-based indexing

#ifdef PARDISO_USE_OUT_OF_CORE_MODE
     iparm[59] = options.out_of_core_mode;
#endif

     long long ierror = pardiso(nullptr, nullptr, 0LL);

     if (ierror != 0LL) {
          cleanup();
          throw std::runtime_error("failed to factor matrix");
     }

     phase = 33LL;
}

template <typename T>
void PardisoObject<T>::init()
{
     std::fill(std::begin(iparm), std::end(iparm), 0);
     std::fill(std::begin(pt), std::end(pt), nullptr);
}

template <typename T>
void PardisoObject<T>::cleanup()
{
     phase = -1;

     long long ierror = pardiso(nullptr, nullptr, 0LL);

     if (ierror != 0LL) {
          warning_with_id("pardiso:cleanup", "failed to cleanup pardiso solver");
     }
}

template <typename T>
long long PardisoObject<T>::pardiso(T* b, T* x, long long nrhs) const
{
     long long ierror = -1LL;

     const int num_threads_prev = mkl_get_max_threads();

     mkl_set_num_threads(options.number_of_threads);

     pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, &Ax.front(), &Ap.front(), &Ai.front(), nullptr, &nrhs, iparm, &msglvl, b, x, &ierror);

     mkl_set_num_threads(num_threads_prev);

     return ierror;
}

template <typename T>
size_t PardisoObject<T>::byte_size() const
{
     return std::max(iparm[14], iparm[15] + iparm[16]);
}

template <typename T>
dim_vector PardisoObject<T>::dims() const
{
     return dim_vector(n, n);
}

template <typename T>
PardisoObject<T>::~PardisoObject()
{
     cleanup();
}

template <typename T>
bool PardisoObject<T>::solve(DenseMatrixType& b, DenseMatrixType& x, long long sys) const {
     if (b.rows() != n) {
          error_with_id("pardiso:solve", "pardiso: rows(b)=%Ld must be equal to rows(A)=%Ld", static_cast<long long>(b.rows()), n);
          return false;
     }

     assert(b.rows() == x.rows());
     assert(b.columns() == x.columns());

     const auto save_sys = iparm[11];
     
     iparm[11] = sys;

     long long ierror = pardiso(b.fortran_vec(), x.fortran_vec(), b.columns());

     iparm[11] = save_sys;

     if (ierror != 0LL) {
          error_with_id("pardiso:solve", "pardiso solve failed with status %Ld", ierror);
          return false;
     }

     return true;
}

template <typename T>
bool PardisoObject<T>::get_options(const octave_value& ovOptions, PardisoObject::Options& options)
{
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
          const auto ithreads = om_options.seek("number_of_threads");

          if (ithreads != om_options.end()) {
               octave_value ov_threads = om_options.contents(ithreads);

               options.number_of_threads = ov_threads.int_value();
          }
     }

#ifdef PARDISO_USE_OUT_OF_CORE_MODE
     {
          const auto iout_of_core = om_options.seek("out_of_core_mode");

          if (iout_of_core != om_options.end()) {
               octave_value ov_out_of_core = om_options.contents(iout_of_core);

               options.out_of_core_mode = ov_out_of_core.int_value();
          }
     }
#endif

     {
          const auto iordering = om_options.seek("ordering");

          if (iordering != om_options.end()) {
               octave_value ov_ordering = om_options.contents(iordering);

               options.ordering = ov_ordering.int_value();
          }
     }

     {
          const auto iscaling = om_options.seek("scaling");

          if (iscaling != om_options.end()) {
               octave_value ov_scaling = om_options.contents(iscaling);

               options.scaling = ov_scaling.int_value();
          }
     }

     {
          const auto iweighted_matching = om_options.seek("weighted_matching");

          if (iweighted_matching != om_options.end()) {
               octave_value ov_weighted_matching = om_options.contents(iweighted_matching);

               options.weighted_matching = ov_weighted_matching.int_value();
          }
     }

     return true;
}

template<typename T>
octave_value_list PardisoObject<T>::eval(const octave_value_list& args, int nargout)
{
     octave_value_list retval;
     octave_idx_type iarg = 0;
     SparseMatrixType A;
     PardisoObject<T>* pPardiso = nullptr;
     bool bOwnPardiso = false;
     bool bHaveMatrix = false;
     bool bHaveRightHandSide = false;

     if (args(iarg).is_matrix_type()) {
          A = (args(iarg++).*PardisoTraits<T>::sparse_matrix_value)(false);

          if (A.rows() != A.columns()) {
               error_with_id("pardiso:input", "pardiso: matrix A must be square");
               return retval;
          }

          if (A.columns() < 1) {
               error_with_id("pardiso:input", "pardiso: matrix A must have at least one column");
               return retval;
          }

          bHaveMatrix = true;
     } else {
          octave_base_value& oOctaveObj = const_cast<octave_base_value&>(args(iarg++).get_rep());
          pPardiso = dynamic_cast<PardisoObject<T>*>(&oOctaveObj);

          if (!pPardiso) {
               error_with_id("pardiso:input", "pardiso: class(pardiso_obj) must be equal to \"pardiso\"");
               return retval;
          }
     }

     DenseMatrixType x, b;

     if (!bHaveMatrix || args.length() >= 3) {
          if (args(0).isreal() && !args(0).is_matrix_type() && args(1).iscomplex()) {
               error_with_id("pardiso:input", "pardiso: complex right hand side cannot be used with a real matrix");
               return retval;
          }

          b = (args(iarg++).*PardisoTraits<T>::matrix_value)(false);
          x.resize(b.rows(), b.columns());

          bHaveRightHandSide = true;
     }

     if (bHaveMatrix) {
          if (args.length() <= iarg) {
               error_with_id("pardiso:input", "pardiso: missing argument \"options\"");
               return retval;
          }

          Options options;

          if (!get_options(args(iarg++), options)) {
               return retval;
          }

          try {
               pPardiso = new PardisoObjectType{A, options};
          } catch(const std::exception& err) {
               error_with_id("pardiso:factor", "%s", err.what());
               return retval;
          }

          bOwnPardiso = true;
     }

     long long sys = 0LL;

     if (args.length() > iarg) {
          sys = args(iarg++).long_value();
     }
     
     if (bHaveRightHandSide) {
          if (pPardiso->solve(b, x, sys)) {
               retval.append(x);
          }

          if (bOwnPardiso) {
               delete pPardiso;
               pPardiso = nullptr;
          }
     } else {
          retval.append(pPardiso);
     }

     return retval;
}

// PKG_ADD: autoload ("pardiso", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("pardiso", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (pardiso, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{pardiso_obj} = pardiso(@var{A}, @var{options})\n\n"
           "@var{x} = pardiso(@var{pardiso_obj}, @var{b})\n\n"
           "@var{x} = pardiso(@var{A}, @var{b}, @var{options})\n\n"
           "Solve a system of linear equations by means of Pardiso (https://software.intel.com)\n\n"
           "The first form creates an factor object @var{pardiso_obj} from matrix @var{A}\n\n"
           "After the factor object has been created, the second form uses the factor object @var{pardiso_obj} to solve a system of linear equations @var{A} * @var{x} = @var{b}\n\n"
           "The third form solves a system of linear equations @var{A} * @var{x} = @var{b} but no factor object is returned\n\n"
           "Several options are supported in struct @var{options}:\n\n"
           "@var{options}.verbose = {true|false}\n\n"
           "@var{options}.symmetric = {true|false}\n\n"
           "@var{options}.refine_max_iter @dots{} maximum number of iterations for refinement of the solution\n\n"
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
          retval = PardisoObject<std::complex<double> >::eval(args, nargout);
     } else {
          retval = PardisoObject<double>::eval(args, nargout);
     }

     return retval;
}

#define DEFINE_GLOBAL_CONSTANT(CONST,VALUE)                             \
     DEFUN_DLD(PARDISO_##CONST, args, nargout, "id = PARDISO_" #CONST  "()\n") \
     {                                                                  \
          return octave_value(octave_int32(VALUE));                     \
     }
