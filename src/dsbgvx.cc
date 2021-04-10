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

#include <algorithm>
#include <cmath>
#include <octave/oct.h>
#include <octave/f77-fcn.h>

octave_idx_type super_diagonals(const SparseMatrix& A);
Matrix sparse_to_symband(const SparseMatrix& A,octave_idx_type KA);

extern "C" 
{
        F77_DBLE F77_FUNC(dlamch, DLAMCH)(F77_CONST_CHAR_ARG_DECL CMACH);
        F77_RET_T F77_FUNC(dsbgvx, DSBGVX)(F77_CONST_CHAR_ARG_DECL JOBZ, F77_CONST_CHAR_ARG_DECL RANGE, F77_CONST_CHAR_ARG_DECL UPLO, F77_INT& N, F77_INT& KA, F77_INT& KB, F77_DBLE AB[], F77_INT& LDAB, F77_DBLE BB[], F77_INT& LDBB, 
                                           F77_DBLE Q[], F77_INT& LDQ, F77_DBLE& VL, F77_DBLE& VU, F77_INT& IL, F77_INT& IU, F77_DBLE& ABSTOL, F77_INT& M, F77_DBLE W[], F77_DBLE Z[],
                                           F77_INT& LDZ, F77_DBLE WORK[], F77_INT IWORK[], F77_INT IFAIL[], F77_INT& INFO);
}

// PKG_ADD: autoload ("dsbgvx", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("dsbgvx", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (dsbgvx, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {} @var{lambda} = dsbgvx(@var{A}, @var{B}, @var{L}, @var{U}, @var{RANGE})\n"
           "@deftypefnx {} [@var{x}, @var{lambda}] = dsbgvx(@dots{})\n"
           "Computes selected eigenvalues, and optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem @var{A} * @var{x} = @var{lambda} * @var{B} * @var{x}\n\n"
           "@var{A} @dots{} banded symmetric matrix\n\n"
           "@var{B} @dots{} banded symmetric positive definite matrix\n\n"
           "@var{L} @dots{} lower limit for eigenvalues to search for\n\n"
           "@var{U} @dots{} upper limit for eigenvalues to search for\n\n"
           "If @var{RANGE} == 'A', search for all eigenvalues\n\n"
           "If @var{RANGE} == 'V', search for eigenvalues within the interval @var{L}-@var{U}\n\n"
           "If @var{RANGE} == 'I', search for the @var{L}-th through the @var{U}-th eigenvalue\n\n"
           "@end deftypefn\n")
{
        octave_value_list retval;
        const octave_idx_type nargin = args.length();

        if (nargin != 5 || nargout > 2)
        {
                print_usage();
                return retval;
        }

        if (!args(0).is_matrix_type())
        {
                error("A must be a matrix!");
                return retval;
        }
  
        if (!args(1).is_matrix_type())
        {
                error("B must be a matrix");
                return retval;
        }
    
        if (!args(2).is_real_scalar())
        {
                error("L muss ein reeller Skalar sein!");
                return retval;
        }
  
        if ( !args(3).is_real_scalar() )
        {
                error("U muss ein reeller Skalar sein!");
                return retval;
        }
  
        if ( !args(4).is_string() || args(4).length() != 1 )
        {
                error("RANGE muss ein String sein!");
                return retval;
        }
  
        char JOBZ = nargout == 2 ? 'V' : 'N';
  
        const SparseMatrix A = args(0).sparse_matrix_value();
  
        F77_INT N = octave::to_f77_int(A.rows());

        if (A.columns() != N || !A.OV_ISSYMMETRIC())
        {
                error("A must be a symmetric matrix");
                return retval;
        }
        
        const SparseMatrix B = args(1).sparse_matrix_value();

        if (!B.OV_ISSYMMETRIC())
        {
                error("B must be a symmetric matrix");
                return retval;
        }
  
        if (B.rows() != N || B.columns() != N)
        {
                error("B must have the same size like A");
                return retval;
        }
  
        char RANGE = args(4).char_matrix_value()(0);
  
        F77_DBLE VL = 0., VU = 0.;
        F77_INT IL = 0, IU = 0;
  
        F77_INT M = N;
  
        switch ( RANGE )
        {
        case 'A':
                break;
        case 'V':
                VL = args(2).scalar_value();
                VU = args(3).scalar_value();
                break;
        case 'I':
                IL = args(2).int_value();
                IU = args(3).int_value();       
                M = IU - IL + 1;
                break;
        default:
                error("RANGE must be one of 'A', 'V', 'I'");
                return retval;
        }
  
        char UPLO = 'U';
    
        F77_INT KA = octave::to_f77_int(super_diagonals(A));
        F77_INT KB = octave::to_f77_int(super_diagonals(B));
  
        if (KB > KA)
        {
                // dsbgvx.f checks if KB.GT.KA
                // If yes,  INFO = -6 will be returned                
                KA = KB;                
        }
        
        Matrix AB = sparse_to_symband(A, KA);
        Matrix BB = sparse_to_symband(B, KB);
  
        F77_INT LDAB = octave::to_f77_int(AB.rows());
        F77_INT LDBB = octave::to_f77_int(BB.rows());
  
        char CMACH = 'S';
        F77_DBLE ABSTOL = 2 * F77_FUNC(dlamch, DLAMCH)(&CMACH);
  
        F77_INT LDQ = JOBZ == 'V' ? std::max<F77_INT>(1, N) : 1;
  
        OCTAVE_LOCAL_BUFFER(F77_DBLE, Q, LDQ * N);
  
        ColumnVector W(N);
  
        F77_INT LDZ = JOBZ == 'V' ? std::max<F77_INT>(1, N) : 1;
  
        Matrix Z(LDZ, N);
  
        OCTAVE_LOCAL_BUFFER(F77_DBLE, WORK, 7 * N);
        OCTAVE_LOCAL_BUFFER(F77_INT, IWORK, 5 * N);
  
        OCTAVE_LOCAL_BUFFER(F77_INT, IFAIL, M);
  
        F77_INT INFO = -1;
  
        F77_XFCN(dsbgvx, DSBGVX, (&JOBZ, &RANGE, &UPLO, N, KA, KB, AB.fortran_vec(), LDAB, BB.fortran_vec(), LDBB, Q, LDQ, VL, VU, IL, IU, ABSTOL, M, W.fortran_vec(), Z.fortran_vec(), LDZ, WORK, IWORK,  IFAIL, INFO));

#if OCTAVE_MAJOR_VERSION < 5
        if (f77_exception_encountered)
        {
                error("Fortran exception in dsbgvx");
                return retval;
        }
#endif
    
        if (INFO != 0)
        {	
                if (INFO < 0)	
                {
                        error("INFO = %d\nif INFO = -i, the i-th argument had an illegal value!", INFO);            
                }
                else if (INFO <= N)
                {
                        error("INFO = %d\nif INFO = i, then i eigenvectors failed to converge!",INFO);
                }
                else
                {
                        error("INFO = %d\ndsbgvx returned an error code;\n"
                              "i.e., if INFO = N + i, for 1 <= i <= N,\n"
                              "then the leading minor of order i of B is not positive definite.\n"
                              "The  factoriza-tion of B could not be completed and no eigenvalues or eigenvectors were computed.",INFO);
                }
		
                return retval;
        }
  
        Z.resize(LDZ, M);
        W.resize(M);
  
        if (JOBZ == 'V')
        {
                retval.append(Z);
        }
	
        retval.append(W);
  
        return retval;
}

octave_idx_type super_diagonals(const SparseMatrix& A)
{
        octave_idx_type KA = 0;       
	
        for (octave_idx_type j = 0; j < A.columns(); ++j)
        {
                for (octave_idx_type i = A.cidx(j); i < A.cidx(j+1); ++i)
                {
                        octave_idx_type ridx = A.ridx(i);
                        octave_idx_type cidx = j;	       
		
                        if (cidx > ridx)
                        {
                                octave_idx_type K = cidx - ridx;
				
                                if ( K > KA )
                                {
                                        KA = K;
                                }
                        }
                }
        }
	
        return KA;
}

Matrix sparse_to_symband(const SparseMatrix& A, octave_idx_type KA)
{
        octave_idx_type LDBA = KA + 1;
	
        octave_idx_type N = A.columns();
	
        Matrix AB(LDBA, N, 0.);
	
        for (octave_idx_type j = 0; j < N; ++j)	{	
                for (octave_idx_type i = std::max<octave_idx_type>(0, j - KA); i <= j; ++i) {
                        AB(KA + i - j, j) = A(i, j);
                }
        }
			
        return AB;
}

/*

%!test
%! ############################################################################################
%! #                                                  TEST 1
%! ############################################################################################

%! clear all;
%! rand("seed", 0);
%! L = 0.5;
%! U = 0.9;
%! RANGE = 'V';
%! for j=1:10
%! A = sprand(10,10,0.3);
%! A = A + A.';
%! P = symrcm(A);
%! A = A(P,P);
%! R = sprand(10,10,0.3);
%! B = R.' * R + eye(10);
%! P = symrcm(B);
%! B = B(P,P);

%! assert(issymmetric(A,0));
%! assert(isdefinite(B,0));

%! [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%! assert(length(lambda),columns(x));
%! assert(rows(x),rows(B));
%! for i=1:length(lambda)
%!  assert(A*x(:,i),lambda(i)*B*x(:,i),sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%! endfor
%! endfor

%!test
%! ############################################################################################
%! #                                                  TEST 2
%! ############################################################################################

%! clear all;
%! rand("seed", 0);
%! L = 1;
%! U = 3;
%! RANGE = 'I';
%! for j=1:10
%! A = sprand(10,10,0.3);
%! A = A + A.';
%! P = symrcm(A);
%! A = A(P,P);
%! R = sprand(10,10,0.3);
%! B = R.' * R + eye(10);
%! P = symrcm(B);
%! B = B(P,P);

%! assert(issymmetric(A,0));
%! assert(isdefinite(B,0));

%! [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%! assert(length(lambda),columns(x));
%! assert(rows(x),rows(B));
%! for i=1:length(lambda)
%!  assert(A*x(:,i),lambda(i)*B*x(:,i),sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%! endfor
%! endfor

%!test
%! ############################################################################################
%! #                                                  TEST 3
%! ############################################################################################

%! clear all;
%! rand("seed", 0);
%! L = 1;
%! U = 10;
%! RANGE = 'A';
%! for j=1:100
%! A = sprand(10,10,0.3) + eye(10);
%! A = A + A.';
%! P = symrcm(A);
%! A = A(P,P);
%! R = sprand(10,10,0.3) + eye(10);
%! B = R.' * R;
%! P = symrcm(B);
%! B = B(P,P);

%! assert(issymmetric(A,0));
%! assert(isdefinite(B,0));

%! [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%! assert(length(lambda),columns(x));
%! assert(rows(x),rows(B));
%! for i=1:length(lambda)
%!  assert(A*x(:,i),lambda(i)*B*x(:,i),sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%! endfor
%! endfor


%!test
%! ############################################################################################
%! #                                                  TEST 4
%! ############################################################################################

%! clear all;
%! rand("seed", 0);
%! L = 1;
%! U = 3;
%! N = 1000;
%! RANGE = 'I';
%! for j=1:3
%! A = sprand(N,N,0.001) + eye(N);
%! A = A + A.';
%! P = symrcm(A);
%! A = A(P,P);
%! R = sprand(N,N,0.001) + eye(N);
%! B = R.' * R;
%! P = symrcm(B);
%! B = B(P,P);

%! assert(issymmetric(A,0));
%! assert(isdefinite(B,0));

%! [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%! assert(length(lambda),columns(x));
%! assert(rows(x),rows(B));
%! for i=1:length(lambda)
%!  assert(A*x(:,i),lambda(i)*B*x(:,i),eps^0.4*norm(lambda(i)*B*x(:,i)));
%! endfor
%! endfor

*/
