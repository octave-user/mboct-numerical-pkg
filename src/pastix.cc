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

#include <stdexcept>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <octave/oct.h>

#define MPI_COMM_WORLD 0

extern "C" {
#include <pastix.h>
}

class PastixObjectDouble;
class PastixObjectComplex;

template <typename T>
struct PastixTraits;

template <>
struct PastixTraits<double> {
     typedef SparseMatrix SparseMatrixType;
     typedef Matrix DenseMatrixType;
     static constexpr spm_coeftype_t flttype = SpmDouble;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_matrix_value;
     static constexpr auto matrix_value = &octave_value::matrix_value;

     static bool isfinite(double x) {
          return std::isfinite(x);
     }

     typedef PastixObjectDouble PastixObjectType;
     static constexpr bool isreal = true;
     static constexpr bool iscomplex = false;
};

template <>
struct PastixTraits<std::complex<double> > {
     typedef SparseComplexMatrix SparseMatrixType;
     typedef ComplexMatrix DenseMatrixType;
     static constexpr spm_coeftype_t flttype = SpmComplex64;

     static constexpr auto sparse_matrix_value = &octave_value::sparse_complex_matrix_value;
     static constexpr auto matrix_value = &octave_value::complex_matrix_value;

     static bool isfinite(const std::complex<double>& z) {
          return std::isfinite(std::real(z)) && std::isfinite(std::imag(z));
     }

     typedef PastixObjectComplex PastixObjectType;
     static constexpr bool isreal = false;
     static constexpr bool iscomplex = true;
};

template <typename T>
class PastixObject : public octave_base_value {
     typedef typename PastixTraits<T>::SparseMatrixType SparseMatrixType;
     typedef typename PastixTraits<T>::DenseMatrixType DenseMatrixType;
     typedef typename PastixTraits<T>::PastixObjectType PastixObjectType;
public:
     struct Options {
          spm_mtxtype_t matrix_type = SpmGeneral;
          pastix_factotype_t factorization = PastixFactLU;
          pastix_int_t number_of_threads = 1;
          pastix_verbose_t verbose = PastixVerboseNo;
          pastix_compress_when_t compress_when = PastixCompressNever;
          pastix_compress_method_t compress_method = PastixCompressMethodPQRCP;
          pastix_compress_ortho_t compress_ortho = PastixCompressOrthoCGS;
          pastix_int_t compress_min_width = 120;
          pastix_int_t compress_min_height = 20;
          double compress_tolerance = 0.01;
          double compress_min_ratio = 1.;
          int refine_max_iter = 3;
          double epsilon_refinement = -1.;
          bool check_solution = false;
     };

     PastixObject();
     explicit PastixObject(const SparseMatrixType& A, const Options& options);
     virtual ~PastixObject(void);
     virtual size_t byte_size() const;
     virtual dim_vector dims() const;
     bool solve(DenseMatrixType& b, DenseMatrixType& x, pastix_trans_t sys) const;
     static bool get_options(const octave_value& ovOptions, PastixObject::Options& options);
     virtual bool is_constant(void) const{ return true; }
     virtual bool is_defined(void) const{ return true; }
     virtual bool isreal() const { return PastixTraits<T>::isreal; }
     virtual bool iscomplex() const { return PastixTraits<T>::iscomplex; }
     static octave_value_list eval(const octave_value_list& args, int nargout);

private:
     void cleanup();

     template <typename U>
     static U* pastix_malloc(std::size_t size) {
          return reinterpret_cast<U*>(malloc(size * sizeof(U)));
     }

     Options options;
     pastix_int_t ncols = 0;
     pastix_int_t* rows = nullptr;
     pastix_int_t* colptr = nullptr;
     T* avals = nullptr;
     mutable pastix_int_t iparm[IPARM_SIZE] = {0};
     mutable double dparm[DPARM_SIZE] = {0};
     mutable pastix_data_t  *pastix_data = nullptr;

     spmatrix_t spm;
     double normA = 0;
};

class PastixObjectDouble: public PastixObject<double> {
public:
     PastixObjectDouble()=default;

     template <typename... Args>
     PastixObjectDouble(const Args&... args)
          :PastixObject<double>(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

class PastixObjectComplex: public PastixObject<std::complex<double> > {
public:
     PastixObjectComplex()=default;

     template <typename... Args>
     PastixObjectComplex(const Args&... args)
          :PastixObject<std::complex<double> >(args...) {
     }

private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(PastixObjectDouble, "pastixd", "pastixd")
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(PastixObjectComplex, "pastixc", "pastixc")

template <typename T>
PastixObject<T>::PastixObject()
{
     std::memset(&spm, 0, sizeof(spm));
}

template <typename T>
PastixObject<T>::PastixObject(const SparseMatrixType& A, const Options& options)
     :options(options),
      ncols(A.columns())
{
     std::memset(&spm, 0, sizeof(spm));

     const octave_idx_type* const cidx = A.cidx();
     const octave_idx_type* const ridx = A.ridx();
     const T* const data = A.data();

     enum MatrixPattern { MAT_SYM_UPPER,
                          MAT_SYM_LOWER,
                          MAT_FULL,
                          MAT_DIAG } eMatPattern = MAT_DIAG;

     switch (options.matrix_type) {
     case SpmSymmetric:
          for (octave_idx_type j = 0; j < ncols; ++j) {
               for (octave_idx_type i = cidx[j]; i < cidx[j + 1]; ++i) {
                    switch (eMatPattern) {
                    case MAT_DIAG:
                         if (ridx[i] > j) {
                              eMatPattern = MAT_SYM_LOWER;
                         } else if (ridx[i] < j) {
                              eMatPattern = MAT_SYM_UPPER;
                         }
                         break;
                    case MAT_SYM_UPPER:
                         if (ridx[i] > j) {
                              eMatPattern = MAT_FULL;
                              goto exit_mat_pattern;
                         }
                         break;
                    case MAT_SYM_LOWER:
                         if (ridx[i] < j) {
                              eMatPattern = MAT_FULL;
                              goto exit_mat_pattern;
                         }
                         break;
                    default:
                         ;
                    }
               }
          }

     exit_mat_pattern:
          ;

          switch (eMatPattern) {
          case MAT_DIAG:
          case MAT_FULL: // The matrix has been declared as symmetric but the full matrix has been provided.
               // According to Pastix manual only the lower triangular part of a symmetric matrix will be used
               eMatPattern = MAT_SYM_LOWER;
               break;
          default:
               ;
          }
          break;

     default:
          eMatPattern = MAT_FULL;
     }

     octave_idx_type nnz = 0;

     switch (eMatPattern) {
     case MAT_SYM_UPPER:
     case MAT_SYM_LOWER:
          for (octave_idx_type j = 0; j < ncols; ++j) {
               for (octave_idx_type i = cidx[j]; i < cidx[j + 1]; ++i) {
                    bool bInsert;

                    switch (eMatPattern) {
                    case MAT_SYM_UPPER:
                         bInsert = ridx[i] <= j;
                         break;
                    case MAT_SYM_LOWER:
                         bInsert = ridx[i] >= j;
                         break;
                    default:
                         bInsert = false;
                    }

                    if (bInsert) {
                         ++nnz;
                    }
               }
          }
          break;

     default:
          nnz = A.nnz();
     }

     rows = pastix_malloc<pastix_int_t>(nnz);
     colptr = pastix_malloc<pastix_int_t>(ncols + 1);
     avals = pastix_malloc<T>(nnz);

     if (!rows || !colptr || !avals) {
          cleanup();
          throw std::bad_alloc();
     }

     switch (eMatPattern) {
     case MAT_SYM_UPPER:
     case MAT_SYM_LOWER: {
          octave_idx_type idx = 0;

          for (octave_idx_type j = 0; j < ncols; ++j) {
               colptr[j] = idx;

               for (octave_idx_type i = cidx[j]; i < cidx[j + 1]; ++i) {
                    bool bInsert;

                    switch (eMatPattern) {
                    case MAT_SYM_UPPER:
                         bInsert = ridx[i] <= j;
                         break;
                    case MAT_SYM_LOWER:
                         bInsert = ridx[i] >= j;
                         break;
                    default:
                         bInsert = false;
                    }

                    if (bInsert) {
                         rows[idx] = ridx[i];
                         avals[idx] = data[i];
                         ++idx;
                    }
               }
          }

          colptr[ncols] = idx;
     } break;
     default:
          // Copy the full matrix because it has been declared as unsymmetrical
          for (octave_idx_type i = 0; i < nnz; ++i) {
               rows[i] = ridx[i];
          }

          for (octave_idx_type i = 0; i < nnz; ++i) {
               avals[i] = data[i];
          }

          for (octave_idx_type i = 0; i < ncols + 1; ++i) {
               colptr[i] = cidx[i];
          }
     }

     spmInit(&spm);

     switch (eMatPattern) {
     case MAT_SYM_LOWER:
     case MAT_SYM_UPPER:
     case MAT_DIAG:
          spm.mtxtype = SpmSymmetric;
          break;

     default:
          spm.mtxtype = SpmGeneral;
     }

     spm.flttype = PastixTraits<T>::flttype;
     spm.fmttype = SpmCSC;
     spm.nnz = nnz;
     spm.n = ncols;
     spm.dof = 1;
     spm.values = avals;
     spm.rowptr = rows;
     spm.colptr = colptr;

     spmUpdateComputedFields(&spm);

     spmatrix_t spm2;

     int rc = spmCheckAndCorrect(&spm, &spm2);

     if (0 != rc) {
          spmExit(&spm);
          spm = spm2;
     }

     pastixInitParam(iparm, dparm);

     iparm[IPARM_VERBOSE] = options.verbose;
     iparm[IPARM_FACTORIZATION] = options.factorization;
     iparm[IPARM_THREAD_NBR] = options.number_of_threads;
     iparm[IPARM_ITERMAX] = options.refine_max_iter;
     iparm[IPARM_COMPRESS_WHEN] = options.compress_when;
     iparm[IPARM_COMPRESS_METHOD] = options.compress_method;
     iparm[IPARM_COMPRESS_ORTHO] = options.compress_ortho;
     iparm[IPARM_COMPRESS_MIN_WIDTH] = options.compress_min_width;
     iparm[IPARM_COMPRESS_MIN_HEIGHT] = options.compress_min_height;
     dparm[DPARM_COMPRESS_TOLERANCE] = options.compress_tolerance;
     dparm[DPARM_COMPRESS_MIN_RATIO] = options.compress_min_ratio;
     dparm[DPARM_EPSILON_REFINEMENT] = options.epsilon_refinement;

     pastixInit(&pastix_data, MPI_COMM_WORLD, iparm, dparm);

     rc = pastix_task_analyze(pastix_data, &spm);

     if (PASTIX_SUCCESS != rc) {
          error_with_id("pastix:solve", "pastix_task_analyze failed with status %d", rc);
          return;
     }

     normA = spmNorm(SpmFrobeniusNorm, &spm);

     spmScalMatrix(1. / normA, &spm);

     rc = pastix_task_numfact(pastix_data, &spm);

     if (PASTIX_SUCCESS != rc) {
          error_with_id("pastix:solve", "pastix_task_numfact failed with status %d", rc);
          return;
     }
}

template <typename T>
size_t PastixObject<T>::byte_size() const
{
     size_t cb = sizeof(*this);

     pastix_int_t nnz = 0;

     if (colptr) {
          nnz = colptr[ncols] - colptr[0];
     }

     cb += sizeof(*rows) * nnz;
     cb += sizeof(*colptr) * (ncols + 1);
     cb += sizeof(*avals) * nnz;
     cb += sizeof(*avals) * iparm[IPARM_NNZEROS];

     return cb;
}

template <typename T>
dim_vector PastixObject<T>::dims() const
{
     return dim_vector(spm.n, spm.n);
}

template <typename T>
PastixObject<T>::~PastixObject()
{
     cleanup();
}

template <typename T>
void PastixObject<T>::cleanup()
{
     if (pastix_data) {
          pastixFinalize(&pastix_data);
     }

     spmExit(&spm); // will free avals, colptr and rows
}

template <typename T>
bool PastixObject<T>::solve(DenseMatrixType& b, DenseMatrixType& x, pastix_trans_t sys) const {
     if (b.rows() != ncols) {
          error_with_id("pastix:solve", "pastix: rows(b)=%ld must be equal to rows(A)=%ld", long(b.rows()), long(ncols));
          return false;
     }

     spmScalVector(spm.flttype, 1. / normA, b.numel(), b.fortran_vec(), 1);

     x = b;

     const auto save_sys = iparm[IPARM_TRANSPOSE_SOLVE];
     
     iparm[IPARM_TRANSPOSE_SOLVE] = sys;
          
     int rc = pastix_task_solve(pastix_data,
                                x.rows(),
                                x.columns(),
                                x.fortran_vec(),
                                x.rows());

     iparm[IPARM_TRANSPOSE_SOLVE] = save_sys;
     
     if (PASTIX_SUCCESS != rc) {
          error_with_id("pastix:solve", "pastix_task_solve failed with status %d", rc);
          return false;
     }

     OCTAVE_QUIT;

     if (options.refine_max_iter) {
          for (octave_idx_type j = 0; j < x.columns(); ++j) {
               bool bZeroVec = true;

               for (octave_idx_type k = 0; k < x.rows(); ++k) {
                    if (x(k, j) != 0.) {
                         bZeroVec = false;
                         break;
                    }
               }

               if (!bZeroVec) {
                    // Avoid division zero by zero in PaStiX
                    rc = pastix_task_refine(pastix_data,
                                            spm.n,
                                            1,
                                            b.fortran_vec() + j * b.rows(),
                                            b.rows(),
                                            x.fortran_vec() + j * x.rows(),
                                            x.rows());

                    if (PASTIX_SUCCESS != rc) {
                         error_with_id("pastix:solve", "pastix_task_refine failed with status %d", rc);
                         return false;
                    }
               }

               OCTAVE_QUIT;
          }

          if (options.check_solution) {
               rc = spmCheckAxb(dparm[DPARM_EPSILON_REFINEMENT],
                                b.columns(),
                                &spm,
                                nullptr,
                                b.rows(),
                                b.fortran_vec(),
                                b.rows(),
                                x.fortran_vec(),
                                x.rows());

               if (SPM_SUCCESS != rc) {
                    error_with_id("pastix:solve", "spmCheckAxb failed with status %d", rc);
                    return false;
               }
          }
     }

     for (octave_idx_type j = 0; j < x.columns(); ++j) {
          for (octave_idx_type i = 0; i < x.rows(); ++i) {
               if (!PastixTraits<T>::isfinite(x(i, j))) {
                    error_with_id("pastix:solve", "solution of pastix is not finite");
                    return false;
               }
          }
     }

     return true;
}

template <typename T>
bool PastixObject<T>::get_options(const octave_value& ovOptions, PastixObject::Options& options)
{
     const octave_scalar_map om_options = ovOptions.scalar_map_value();

#if OCTAVE_MAJOR_VERSION < 6
     if (error_state) {
          return false;
     }
#endif

     {
          const auto imat_type = om_options.seek("matrix_type");

          if (imat_type != om_options.end()) {
               options.matrix_type = static_cast<spm_mtxtype_t>(om_options.contents(imat_type).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto ifactor = om_options.seek("factorization");

          if (ifactor != om_options.end()) {
               options.factorization = static_cast<pastix_factotype_t>(om_options.contents(ifactor).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto inum_threads = om_options.seek("number_of_threads");

          if (inum_threads != om_options.end()) {
               options.number_of_threads = om_options.contents(inum_threads).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto iverbose = om_options.seek("verbose");

          if (iverbose != om_options.end()) {
               options.verbose = static_cast<pastix_verbose_t>(om_options.contents(iverbose).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto irefine = om_options.seek("refine_max_iter");

          if (irefine != om_options.end()) {
               octave_value ov_ref = om_options.contents(irefine);

               options.refine_max_iter = ov_ref.int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }

     {
          const auto icheck = om_options.seek("check_solution");
          if (icheck != om_options.end()) {
               options.check_solution = om_options.contents(icheck).bool_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }

     {
          const auto icompress_when = om_options.seek("compress_when");

          if (icompress_when != om_options.end()) {
               options.compress_when = static_cast<pastix_compress_when_t>(om_options.contents(icompress_when).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto icompress_method = om_options.seek("compress_method");

          if (icompress_method != om_options.end()) {
               options.compress_method = static_cast<pastix_compress_method_t>(om_options.contents(icompress_method).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto icompress_ortho = om_options.seek("compress_ortho");

          if (icompress_ortho != om_options.end()) {
               options.compress_ortho = static_cast<pastix_compress_ortho_t>(om_options.contents(icompress_ortho).int_value());

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }

     {
          const auto icompress_min_width = om_options.seek("compress_min_width");

          if (icompress_min_width != om_options.end()) {
               options.compress_min_width = om_options.contents(icompress_min_width).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto icompress_min_height = om_options.seek("compress_min_height");

          if (icompress_min_height != om_options.end()) {
               options.compress_min_height = om_options.contents(icompress_min_height).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto icompress_tolerance = om_options.seek("compress_tolerance");

          if (icompress_tolerance != om_options.end()) {
               options.compress_tolerance = om_options.contents(icompress_tolerance).scalar_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }
     {
          const auto icompress_min_ratio = om_options.seek("compress_min_ratio");

          if (icompress_min_ratio != om_options.end()) {
               options.compress_min_ratio = om_options.contents(icompress_min_ratio).scalar_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }

     {
          const auto iepsilon_refinement = om_options.seek("epsilon_refinement");

          if (iepsilon_refinement != om_options.end()) {
               options.epsilon_refinement = om_options.contents(iepsilon_refinement).scalar_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return false;
               }
#endif
          }
     }

     return true;
}

template<typename T>
octave_value_list PastixObject<T>::eval(const octave_value_list& args, int nargout)
{
     octave_value_list retval;
     octave_idx_type iarg = 0;
     SparseMatrixType A;
     PastixObject<T>* pPastix = nullptr;
     bool bOwnPastix = false;
     bool bHaveMatrix = false;
     bool bHaveRightHandSide = false;

     if (args(iarg).is_matrix_type()) {
          A = (args(iarg++).*PastixTraits<T>::sparse_matrix_value)(false);

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               return retval;
          }
#endif

          if (A.rows() != A.columns()) {
               error_with_id("pastix:input", "pastix: matrix A must be square");
               return retval;
          }

          if (A.columns() < 1) {
               error_with_id("pastix:input", "pastix: matrix A must have at least one column");
               return retval;
          }

          bHaveMatrix = true;
     } else {
          octave_base_value& oOctaveObj = const_cast<octave_base_value&>(args(iarg++).get_rep());
          pPastix = dynamic_cast<PastixObject<T>*>(&oOctaveObj);

          if (!pPastix) {
               error_with_id("pastix:input", "pastix: class(pastix_obj) must be equal to \"pastix\"");
               return retval;
          }
     }

     DenseMatrixType x, b;

     if (!bHaveMatrix || args.length() >= 3) {
          if (args(0).isreal() && !args(0).is_matrix_type() && args(1).iscomplex()) {
               error_with_id("pastix:input", "pastix: complex right hand side cannot be used with a real matrix");
               return retval;
          }

          b = (args(iarg++).*PastixTraits<T>::matrix_value)(false);

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               return retval;
          }
#endif

          bHaveRightHandSide = true;
     }

     if (bHaveMatrix) {
          if (args.length() <= iarg) {
               error_with_id("pastix:input", "pastix: missing argument \"options\"");
               return retval;
          }

          Options options;

          if (!get_options(args(iarg++), options)) {
               return retval;
          }

          pPastix = new PastixObjectType{A, options};

          bOwnPastix = true;

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               delete pPastix;
               return retval;
          }
#endif
     }

     pastix_trans_t sys = PastixNoTrans;

     if (args.length() > iarg) {
          switch (args(iarg++).long_value()) {
          case 0:
               sys = PastixNoTrans;
               break;
          case 1:
               sys = PastixConjTrans;
               break;
          case 2:
               sys = PastixTrans;
               break;
          default:
               sys = static_cast<pastix_trans_t>(-1);
          }
     }
     
     if (bHaveRightHandSide) {
          if (pPastix->solve(b, x, sys)) {
               retval.append(x);
          }

          if (bOwnPastix) {
               delete pPastix;
               pPastix = nullptr;
          }
     } else {
          retval.append(pPastix);
     }

     return retval;
}

// PKG_ADD: autoload ("pastix", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("pastix", "__mboct_numerical__.oct", "remove");

// PKG_ADD: autoload ("PASTIX_API_SYM_YES", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_SYM_NO", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_FACT_LLT", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_FACT_LDLT", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_FACT_LU", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_FACT_LLH", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_FACT_LDLH", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_VERBOSE_NOT", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_VERBOSE_NO", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_VERBOSE_YES", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_BIND_NO", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("PASTIX_API_BIND_AUTO", "__mboct_numerical__.oct");

// PKG_DEL: autoload ("PASTIX_API_SYM_YES", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_SYM_NO", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_FACT_LLT", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_FACT_LDLT", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_FACT_LU", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_FACT_LLH", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_FACT_LDLH", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_VERBOSE_NOT", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_VERBOSE_NO", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_VERBOSE_YES", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_BIND_NO", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("PASTIX_API_BIND_AUTO", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (pastix, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{pastix_obj} = pastix(@var{A}, @var{options})\n\n"
           "@var{x} = pastix(@var{pastix_obj}, @var{b})\n\n"
           "@var{x} = pastix(@var{A}, @var{b}, @var{options})\n\n"
           "Solve a system of linear equations by means of PaStiX (http://pastix.gforge.inria.fr/)\n\n"
           "The first form creates an factor object @var{pastix_obj} from matrix @var{A}\n\n"
           "After the factor object has been created, the second form uses the factor object @var{pastix_obj} to solve a system of linear equations @var{A} * @var{x} = @var{b}\n\n"
           "The third form solves a system of linear equations @var{A} * @var{x} = @var{b} but no factor object is returned\n\n"
           "Several options are supported in struct @var{options}:\n\n"
           "@var{options}.verbose = PASTIX_API_VERBOSE_NOT | PASTIX_API_VERBOSE_NO | PASTIX_API_VERBOSE_YES\n\n"
           "@var{options}.factorization = PASTIX_API_FACT_LU | PASTIX_API_FACT_LLT | PASTIX_API_FACT_LDLT\n\n"
           "@var{options}.matrix_type = PASTIX_API_SYM_NO | PASTIX_API_SYM_YES\n\n"
           "@var{options}.refine_max_iter @dots{} maximum number of iterations for refinement of the solution\n\n"
           "@var{options}.bind_thread_mode = PASTIX_API_BIND_NO | PASTIX_API_BIND_AUTO\n\n"
           "@var{options}.number_of_threads @dots{} number of threads to use\n\n"
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
          retval = PastixObject<std::complex<double> >::eval(args, nargout);
     } else {
          retval = PastixObject<double>::eval(args, nargout);
     }

     return retval;
}

#define DEFINE_GLOBAL_CONSTANT(CONST,VALUE)                             \
     DEFUN_DLD(PASTIX_##CONST, args, nargout, "id = PASTIX_" #CONST  "()\n") \
     {									\
          return octave_value(octave_int32(VALUE));			\
     }

#define DEFINE_GLOBAL_CONSTANT2(CONST)          \
     DEFINE_GLOBAL_CONSTANT(CONST,CONST)

DEFINE_GLOBAL_CONSTANT(API_SYM_YES, SpmSymmetric)
DEFINE_GLOBAL_CONSTANT(API_SYM_NO, SpmGeneral)
DEFINE_GLOBAL_CONSTANT(API_FACT_LLT, PastixFactLLT)
DEFINE_GLOBAL_CONSTANT(API_FACT_LDLT, PastixFactLDLT)
DEFINE_GLOBAL_CONSTANT(API_FACT_LU, PastixFactLU)
DEFINE_GLOBAL_CONSTANT(API_FACT_LDLH, PastixFactLDLH)
DEFINE_GLOBAL_CONSTANT(API_FACT_LLH, PastixFactLLH)
DEFINE_GLOBAL_CONSTANT(API_VERBOSE_NOT, PastixVerboseNot)
DEFINE_GLOBAL_CONSTANT(API_VERBOSE_NO, PastixVerboseNo)
DEFINE_GLOBAL_CONSTANT(API_VERBOSE_YES, PastixVerboseYes)
DEFINE_GLOBAL_CONSTANT(API_BIND_NO, -1)
DEFINE_GLOBAL_CONSTANT(API_BIND_AUTO, -2)
