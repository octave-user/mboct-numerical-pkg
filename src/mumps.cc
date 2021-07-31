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

#include <stdexcept>
#include <cstring>
#include <complex>
#include <octave/oct.h>
#include <octave/dSparse.h>
#include <octave/CSparse.h>

extern "C" {
#ifdef USE_MUMPS_SEQ_MPI_H
#include <mumps_seq/mpi.h>
#endif
#include <dmumps_c.h>
#include <zmumps_c.h>
}

template <typename T>
struct MumpsTraits;

template <>
struct MumpsTraits<double> {
     typedef DMUMPS_STRUC_C MUMPS_STRUC_C;
     typedef SparseMatrix SparseMatrixType;
     typedef Matrix DenseMatrixType;
     typedef DMUMPS_REAL ScalarType;

     static void mumps_c(MUMPS_STRUC_C* id) {
          dmumps_c(id);
     }

     static SparseMatrixType sparse_matrix_value(const octave_value& ov) {
          return ov.sparse_matrix_value();
     }

     static DenseMatrixType dense_matrix_value(const octave_value& ov) {
          return ov.matrix_value();
     }

     static ScalarType* convert_pointer(double* b) {
          return b;
     }

     static constexpr bool isreal = true;
};

template <>
struct MumpsTraits<std::complex<double> > {
     typedef ZMUMPS_STRUC_C MUMPS_STRUC_C;
     typedef SparseComplexMatrix SparseMatrixType;
     typedef ComplexMatrix DenseMatrixType;
     typedef ZMUMPS_COMPLEX ScalarType;

     static void mumps_c(MUMPS_STRUC_C* id) {
          zmumps_c(id);
     }

     static SparseMatrixType sparse_matrix_value(const octave_value& ov) {
          return ov.sparse_complex_matrix_value();
     }

     static DenseMatrixType dense_matrix_value(const octave_value& ov) {
          return ov.complex_matrix_value();
     }

     static ScalarType* convert_pointer(std::complex<double>* b) {
          static_assert(sizeof(std::complex<double>) == sizeof(ScalarType));

          return reinterpret_cast<ScalarType*>(b);
     }
     
     static constexpr bool isreal = false;
};

class MumpsBase: public octave_base_value {
public:
     enum MatrixType {
          MAT_GEN = 0,
          MAT_DEF = 1,
          MAT_SYM = 2
     };

     enum VerboseType {
          VER_NO = 0,
          VER_ERR = 1,
          VER_WARN = 2,
          VER_DIAG = 3,
          VER_ALL = 4
     };

     struct Options {
          MatrixType matrix_type = MAT_GEN;
          VerboseType verbose = VER_NO;
          int refine_max_iter = 250;
          int workspace_inc = -1;
          double epsilon_refinement = -1.;
     };
};

template <typename T>
class MumpsObject : public MumpsBase {
public:
     typedef typename MumpsTraits<T>::MUMPS_STRUC_C MUMPS_STRUC_C;
     typedef typename MumpsTraits<T>::SparseMatrixType SparseMatrixType;
     typedef typename MumpsTraits<T>::DenseMatrixType DenseMatrixType;
     typedef typename MumpsTraits<T>::ScalarType ScalarType;

     MumpsObject();
     explicit MumpsObject(const SparseMatrixType& A, const Options& options);
     virtual ~MumpsObject(void);
     DenseMatrixType solve(const DenseMatrixType& b);
     static octave_value_list solve(const octave_value_list& args);
     virtual bool is_constant(void) const override { return true; }
     virtual bool is_defined(void) const override { return true; }
     virtual dim_vector dims (void) const override { return dim_vector (id.n, id.n); }
     virtual bool isreal() const override { return MumpsTraits<T>::isreal; }
     virtual bool iscomplex() const override { return !MumpsTraits<T>::isreal; }     
     virtual void print(std::ostream& os, bool pr_as_read_syntax) override;

private:
     void cleanup();

     Array<MUMPS_INT> irn, jcn;
     Array<T> a;
     MUMPS_STRUC_C id;
     int idmpi;
     Options options;
};

class DMumpsObject: public MumpsObject<double> {
public:
     DMumpsObject() {
     }

     DMumpsObject(const SparseMatrix& A, const Options& options)
          :MumpsObject<double>(A, options) {
     }
private:
     DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(DMumpsObject, "dmumps", "dmumps");

template <typename T>
MumpsObject<T>::MumpsObject()
     :idmpi(-1) {
     std::memset(&id, 0, sizeof(id));
}

template <typename T>
MumpsObject<T>::MumpsObject(const SparseMatrixType& A, const Options& opt)
     : irn(dim_vector(A.nnz(), 1)),
       jcn(dim_vector(A.nnz(), 1)),
       a(dim_vector(A.nnz(), 1)),
       options(opt) {

     std::memset(&id, 0, sizeof(id));

     if (A.rows() != A.columns()) {
          throw std::runtime_error("matrix A must be square");
     }

     const octave_idx_type n = A.columns();
     octave_idx_type nz = A.nnz();
     const octave_idx_type* const cidx = A.cidx();
     const octave_idx_type* const ridx = A.ridx();
     const T* const data = A.data();
     octave_idx_type idx = 0;

     enum MatrixPattern { MAT_SYM_UPPER,
                          MAT_SYM_LOWER,
                          MAT_FULL,
                          MAT_DIAG } eMatPattern = MAT_DIAG;

     switch (options.matrix_type) {
     case MAT_DEF:
     case MAT_SYM:
          for (octave_idx_type j = 0; j < n; ++j) {
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
               eMatPattern = MAT_SYM_LOWER;
               break;
          default:
               ;
          }
          break;

     default:
          eMatPattern = MAT_FULL;
     }

     for (octave_idx_type j = 0; j < n; ++j) {
          for (octave_idx_type i = cidx[j]; i < cidx[j + 1]; ++i) {
               assert(idx < nz);
               octave_idx_type r = ridx[i] + 1;
               octave_idx_type c = j + 1;

               bool bCopy;

               switch (eMatPattern) {
               case MAT_SYM_UPPER:
                    bCopy = r <= c;
                    break;
               case MAT_SYM_LOWER:
                    bCopy = r >= c;
                    break;
               default:
                    bCopy = true;
               }

               if (bCopy) {
                    irn(idx) = r;
                    jcn(idx) = c;
                    a.xelem(idx) = data[i];
                    ++idx;
               }
          }
     }

     assert(idx <= nz);

     nz = idx;

     irn.resize(dim_vector(nz, 1));
     jcn.resize(dim_vector(nz, 1));
     a.resize(dim_vector(nz, 1));

#ifdef USE_MUMPS_SEQ_MPI_H
     {
          int ierr = MPI_Init(nullptr, nullptr);

          ierr = MPI_Comm_rank(MPI_COMM_WORLD, &idmpi);
     }
#endif
     
     id.job = -1;
     id.par = 1;
     id.sym = options.matrix_type;
     
#ifdef USE_MUMPS_SEQ_MPI_H
     id.comm_fortran = MPI_COMM_WORLD;
#else
     id.comm_fortran = 0;
#endif
     
     MumpsTraits<T>::mumps_c(&id);

     if (id.info[0] != 0) {
          throw std::runtime_error("mumps: faild to initialize");
     }

     id.n = n;
     id.nz = nz;
     id.irn = irn.fortran_vec();
     id.jcn = jcn.fortran_vec();
     id.a = MumpsTraits<T>::convert_pointer(a.fortran_vec());

     for (int i = 1; i <= 3; ++i) {
          if (options.verbose < i) {
               id.icntl[i - 1] = -1;
          }
     }

     id.icntl[4 - 1] = options.verbose;
     id.icntl[10 - 1] = options.refine_max_iter;

     if (options.workspace_inc >= 0) {
          id.icntl[14 - 1] = options.workspace_inc;
     }

     id.cntl[2 - 1] = options.epsilon_refinement;


     id.job = 1;

     MumpsTraits<T>::mumps_c(&id);

     if (id.info[0] != 0) {
          throw std::runtime_error("mumps: failed to analyze matrix");
     }

     OCTAVE_QUIT;

     id.job = 2;

     MumpsTraits<T>::mumps_c(&id);

     if (id.info[0] != 0) {
          throw std::runtime_error("mumps: failed to factor matrix");
     }

     OCTAVE_QUIT;
}

template <typename T>
MumpsObject<T>::~MumpsObject()
{
     if (id.job != 0) {
          id.job = -2;
          MumpsTraits<T>::mumps_c(&id);
     }
#ifdef USE_MUMPS_SEQ_MPI_H
     MPI_Finalize();
#endif
}

template <typename T>
typename MumpsObject<T>::DenseMatrixType
MumpsObject<T>::solve(const DenseMatrixType& b)
{
     DenseMatrixType x = b;

     T* px = x.fortran_vec();

     id.nrhs = 1;
     id.lrhs = x.rows();
     id.job = 3;

     for (octave_idx_type i = 0; i < x.columns(); ++i) {
          id.rhs = MumpsTraits<T>::convert_pointer(px);

          MumpsTraits<T>::mumps_c(&id);

          if (id.info[0] != 0) {
               throw std::runtime_error("mumps: failed to solve for x");
          }

          px += x.rows();
     }

     return x;
}

template <typename T>
octave_value_list
MumpsObject<T>::solve(const octave_value_list& args)
{
     octave_value_list retval;
     octave_idx_type iarg = 0;
     SparseMatrixType A;
     MumpsObject<T>* pMumps = nullptr;
     bool bOwnMumps = false;
     bool bHaveMatrix = false;
     bool bHaveRightHandSide = false;
     octave_idx_type n = 0;

     if (args(iarg).is_matrix_type()) {
          A = MumpsTraits<T>::sparse_matrix_value(args(iarg++));

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               return retval;
          }
#endif

          if (A.rows() != A.columns()) {
               error_with_id("mumps:input", "pastix: matrix A must be square");
               return retval;
          }

          if (A.columns() < 1) {
               error_with_id("mumps:input", "mumps: matrix A must have at least one column");
               return retval;
          }

          n = A.rows();

          bHaveMatrix = true;
     } else {
          octave_base_value& oOctaveObj = const_cast<octave_base_value&>(args(iarg++).get_rep());
          pMumps = dynamic_cast<MumpsObject<T>*>(&oOctaveObj);

          if (!pMumps) {
               error_with_id("mumps:input", "mumps: class(mumps_obj) must be equal to \"mumps\"");
               return retval;
          }

          n = pMumps->rows();
     }

     DenseMatrixType b;

     if (!bHaveMatrix || args.length() > 2) {
          if (args.length() <= iarg) {
               print_usage();
               return retval;
          }

          b = MumpsTraits<T>::dense_matrix_value(args(iarg++));

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               return retval;
          }
#endif

          if (b.rows() != n) {
               error_with_id("mumps:input", "mumps: number of rows of b does not match number of rows of A");
               return retval;
          }

          if (b.columns() < 1) {
               error_with_id("mumps:input", "number of columns of b must be at least one");
               return retval;
          }

          bHaveRightHandSide = true;
     }

     MumpsObject::Options options;

     if (bHaveMatrix) {
          if (args.length() <= iarg) {
               print_usage();
               return retval;
          }

          octave_scalar_map ov_options = args(iarg++).scalar_map_value();

#if OCTAVE_MAJOR_VERSION < 6
          if (error_state) {
               return retval;
          }
#endif

          auto iter_mat_type = ov_options.seek("matrix_type");

          if (iter_mat_type != ov_options.end()) {
               int mat_type = ov_options.contents(iter_mat_type).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return retval;
               }
#endif

               options.matrix_type = static_cast<MumpsObject::MatrixType>(mat_type);
          }

          auto iter_verbose = ov_options.seek("verbose");

          if (iter_verbose != ov_options.end()) {
               int verbose = ov_options.contents(iter_verbose).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return retval;
               }
#endif
               options.verbose = static_cast<MumpsObject::VerboseType>(verbose);
          }

          auto iter_refine_max_iter = ov_options.seek("refine_max_iter");

          if (iter_refine_max_iter != ov_options.end()) {
               options.refine_max_iter = ov_options.contents(iter_refine_max_iter).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return retval;
               }
#endif
          }

          auto iter_epsilon_refinement = ov_options.seek("epsilon_refinement");

          if (iter_epsilon_refinement != ov_options.end()) {
               options.epsilon_refinement = ov_options.contents(iter_epsilon_refinement).scalar_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return retval;
               }
#endif
          }

          auto iter_workspace_inc = ov_options.seek("workspace_inc");

          if (iter_workspace_inc != ov_options.end()) {
               options.workspace_inc = ov_options.contents(iter_workspace_inc).int_value();

#if OCTAVE_MAJOR_VERSION < 6
               if (error_state) {
                    return retval;
               }
#endif
          }
     }

     try {
          if (bHaveMatrix) {
               pMumps = new MumpsObject<T>{A, options};

               bOwnMumps = true;
          }

          if (bHaveRightHandSide) {
               retval.append(pMumps->solve(b));

               if (bOwnMumps) {
                    delete pMumps;
                    pMumps = nullptr;
               }
          } else {
               retval.append(pMumps);
          }
     } catch (const std::exception& err) {
          if (bOwnMumps) {
               delete pMumps;
               pMumps = nullptr;
          }

          error_with_id("mumps:solver", "%s", err.what());
     }

     return retval;
}

template <typename T>
void MumpsObject<T>::print(std::ostream& os, bool pr_as_read_syntax) {
     os << "mumps(n=" << id.n
        << " nz=" << id.nz
        << " sym=" << id.sym
        << ")" << std::endl;
}

// PKG_ADD: autoload ("mumps", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_MAT_GEN", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_MAT_DEF", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_MAT_SYM", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_VER_NO", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_VER_ERR", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_VER_WARN", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_VER_DIAG", "__mboct_numerical__.oct");
// PKG_ADD: autoload ("MUMPS_VER_ALL", "__mboct_numerical__.oct");

// PKG_DEL: autoload ("mumps", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_MAT_GEN", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_MAT_DEF", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_MAT_SYM", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_VER_NO", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_VER_ERR", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_VER_WARN", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_VER_DIAG", "__mboct_numerical__.oct", "remove");
// PKG_DEL: autoload ("MUMPS_VER_ALL", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (mumps, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{mumps_obj} = mumps(@var{A}, @var{opt})\n\n"
           "@var{x} = mumps(@var{mumps_obj}, @var{b})\n\n"
           "@var{x} = mumps(@var{A}, @var{b}, @var{opt})\n\n"
           "Solve a system of linear equations by means of MUMPS (http://mumps.enseeiht.fr/)\n\n"
           "The first form creates an factor object @var{mumps_obj} from matrix @var{A}\n\n"
           "After the factor object has been created, the second form uses the factor object @var{mumps_obj} to solve a system of linear equations @var{A} * @var{x} = @var{b}\n\n"
           "The third form solves a system of linear equations @var{A} * @var{x} = @var{b} but no factor object is returned\n\n"
           "Several options are supported in struct @var{opt}:\n\n"
           "@var{opt}.verbose = 0:4\n\n"
           "@var{opt}.matrix_type = 0:2\n\n"
           "@var{opt}.refine_max_iter @dots{} maximum number of iterations for refinement of the solution\n\n"
           "@var{opt}.workspace_inc @dots{} maximum percentage of increase of workspace size during factorization\n\n"
           "@end deftypefn\n")
{
     if (args.length() < 1 || nargout > 1) {
          print_usage();
          return octave_value_list();
     }

     if (args(0).isreal()) {
          return MumpsObject<double>::solve(args);
     } else if (args(0).iscomplex()) {
          return MumpsObject<std::complex<double> >::solve(args);
     } else {
          error_with_id("mumps:solver", "A must be a real or complex matrix");
          return octave_value_list();
     }
}


#define DEFINE_GLOBAL_CONSTANT(CONST)                                   \
     DEFUN_DLD(MUMPS_##CONST, args, nargout, "id = MUMPS_" #CONST  "()\n") \
     {                                                                  \
          return octave_value(octave_int32(MumpsBase::CONST));          \
     }

DEFINE_GLOBAL_CONSTANT(MAT_GEN)
DEFINE_GLOBAL_CONSTANT(MAT_DEF)
DEFINE_GLOBAL_CONSTANT(MAT_SYM)
DEFINE_GLOBAL_CONSTANT(VER_NO)
DEFINE_GLOBAL_CONSTANT(VER_ERR)
DEFINE_GLOBAL_CONSTANT(VER_WARN)
DEFINE_GLOBAL_CONSTANT(VER_DIAG)
DEFINE_GLOBAL_CONSTANT(VER_ALL)

/*
%!error mumps([]);
%!error mumps(eye(3));
%!error mumps(eye(3), zeros(3,1));
%!error mumps(mumps(eye(3), struct()));
%!error mumps(eye(3),zeros(2,1),struct());
%!error mumps(ones(3,2),zeros(3,1),struct());
%!test
%! rand("seed", 0);
%! n = [2,4,8,16,32,64,128];
%! for e=[0,100]
%!   for u=1:2
%!     for m=[MUMPS_MAT_GEN, MUMPS_MAT_DEF, MUMPS_MAT_SYM]
%!       for i=1:10
%!         for j=1:numel(n)
%!           for k=1:2
%!             for l=1:2
%!               for s=1:2
%!                 switch (l)
%!                   case 1
%!                     A = rand(n(j),n(j));
%!                   otherwise
%!                     A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                 endswitch
%!                 Af = A;
%!                 b = rand(n(j), 3);
%!                 switch (s)
%!                   case {1,2}
%!                     switch (s)
%!                       case 2
%!                         A += A.';
%!                       case 1
%!                         A *= A.';
%!                         A += eye(size(A));
%!                     endswitch
%!                     Af = A;
%!                     if (s == 2 && m > 0)
%!                       [r, c, d] = find(A);
%!                       if (u == 1)
%!                         idx = find(r >= c);
%!                       else
%!                         idx = find(r <= c);
%!                       endif
%!                       r = r(idx);
%!                       c = c(idx);
%!                       d = d(idx);
%!                       Af = sparse(r, c, d, n(j), n(j));
%!                       if (l == 1)
%!                         Af = full(Af);
%!                       endif
%!                     endif
%!                 endswitch
%!                 xref = A \ b;
%!                 opt.verbose = MUMPS_VER_WARN;
%!                 opt.refine_max_iter = e;
%!                 opt.matrix_type = m;
%!                 switch (k)
%!                   case 1
%!                     x = mumps(Af, b, opt);
%!                   otherwise
%!                     x = mumps(mumps(Af, opt), b);
%!                 endswitch
%!                 assert(A * x, b, sqrt(eps) * norm(b));
%!                 assert(norm(A * x - b) < sqrt(eps) * norm(A*x+b));
%!                 assert(x, xref, sqrt(eps) * norm(x));
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endfor
*/
