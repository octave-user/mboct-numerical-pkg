## Copyright (C) 2011(-2020) Reinhard <octave-user@a1.net>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## This package contains interfaces to several well known numerical solvers like pastix, mumps, umfpack and arpack and metis.

%!test
%! if (numel(which("pastix")))
%! for i=1:2
%! for j=1:2
%! A = [1 0 0 0 0
%! 0 3 0 0 0
%! 2 0 5 0 0
%! 0 0 6 7 0
%! 0 0 0 0 8];
%! b = [1, 9;
%!      2, 5;
%!      3, 4;
%!      4, 8;
%!      6, 7];
%! switch (j)
%! case 2
%!  A += 0.5j * A;
%!  b += 0.5j * b;
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.factorization = PASTIX_API_FACT_LU;
%! opts.matrix_type = PASTIX_API_SYM_NO;
%! opts.refine_max_iter = int32(10);
%! opts.bind_thread_mode = PASTIX_API_BIND_NO;
%! opts.number_of_threads = int32(4);
%! opts.check_solution = true;
%! switch i
%! case 1
%! x = pastix(A, b, opts);
%! case 2
%! x = pastix(pastix(A, opts), b);
%! endswitch
%! f = A * x - b;
%! assert(norm(f) < eps^0.8 * norm(b));
%! assert(x, A \ b, eps^0.8 * norm(A \ b));
%! endfor
%! endfor
%! else
%! warning("pastix is not installed");
%! endif

%!test
%! if (numel(which("pastix")))
%! tol = eps^0.35;
%! rand("seed", 0);
%! for N=[2, 10, 100]
%! for i=1:10
%! for j=1:2
%! for k=1:2
%! A = rand(N, N);
%! b = rand(N, 10);
%! switch (k)
%! case 2
%! A += 1j * rand(N, N);
%! b += 1j * rand(N, 10);
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.factorization = PASTIX_API_FACT_LU;
%! opts.matrix_type = PASTIX_API_SYM_NO;
%! opts.refine_max_iter = int32(10);
%! opts.check_solution = true;
%! switch j
%! case 1
%! x = pastix(A, b, opts);
%! case 2
%! x = pastix(pastix(A, opts), b);
%! endswitch
%! f = A * x - b;
%! assert(all(norm(f, "cols") < tol * norm(b, "cols")));
%! assert(x, A \ b, tol * norm(A \ b, "cols"));
%! endfor
%! endfor
%! endfor
%! endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! tol = eps^0.35;
%! rand("seed", 0);
%! for N=[2, 10, 100]
%! for i=1:10
%! for j=1:2
%! for k=1:2
%! A = rand(N, N);
%! b = rand(N, 10);
%! switch (k)
%! case 2
%! A += 1j * rand(N, N);
%! b += 1j * rand(N, 10);
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.factorization = PASTIX_API_FACT_LU;
%! opts.matrix_type = PASTIX_API_SYM_NO;
%! opts.refine_max_iter = int32(10);
%! opts.check_solution = true;
%! switch j
%! case 1
%! x = pastix(A, b, opts);
%! case 2
%! x = pastix(pastix(A, opts), b);
%! endswitch
%! f = A * x - b;
%! assert(all(norm(f, "cols") < tol * norm(b, "cols")));
%! assert(x, A \ b, tol * norm(A \ b, "cols"));
%! endfor
%! endfor
%! endfor
%! endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! tol = eps^0.3;
%! rand("seed", 0);
%! for ref=int32([10])
%! for bind=[PASTIX_API_BIND_NO]
%! for t=[1,4]
%! for s=0:1
%! for f=[PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT]
%! for N=[10, 100]
%! for i=1:10
%! for j=1:2
%! for k=1:2
%! A = rand(N, N);
%! switch (k)
%! case 2
%! A += 1j * rand(N, N);
%! endswitch
%! A *= A.';
%! [r, c, d] = find(A);
%! if s
%! idx = find(r >= c);
%! else
%! idx = 1:numel(r);
%! endif
%! b = rand(N, 10);
%! switch (k)
%! case 2
%! b += 1j * rand(N, 10);
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.refine_max_iter = ref;
%! opts.matrix_type = PASTIX_API_SYM_YES;
%! opts.factorization = f;
%! opts.number_of_threads = t;
%! opts.bind_thread_mode = bind;
%! opts.check_solution = true;
%! switch j
%! case 1
%! x = pastix(sparse(r(idx), c(idx), d(idx)), b, opts);
%! case 2
%! x = pastix(pastix(sparse(r(idx), c(idx), d(idx)), opts), b);
%! endswitch
%! xref = A \ b;
%! ferr = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%! fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%! assert(max(ferr) < 10 * max(fref));
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! for k=1:2
%! tol = sqrt(eps);
%! A = [ 1, -1,  0, 0,  1;
%!      -1,  2, -1, 0,  0;
%!       0, -1,  2, -1, 0;
%!       0,  0, -1,  1, 0;
%!       1,  0,  0,  0, 0];

%! b = [1; 2; 3; 4; 5];
%! switch (k)
%! case 2
%! A += 0.5j * A;
%! b += 0.5j * b;
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.refine_max_iter = int32(10);
%! opts.matrix_type = PASTIX_API_SYM_YES;
%! opts.factorization = PASTIX_API_FACT_LDLT;
%! opts.check_solution = true;
%! for i=1:2
%! [r, c, d] = find(A);
%! idx = find(r >= c);
%! Asym = sparse(r(idx), c(idx), d(idx), rows(A), columns(A));
%! switch i
%! case 1
%! x = pastix(Asym, b, opts);
%! case 2
%! x = pastix(pastix(Asym, opts), b);
%! endswitch
%! f = A * x - b;
%! assert(all(norm(f, "cols") < tol * norm(b, "cols")));
%! assert(x, A \ b, tol * norm(A \ b, "cols"));
%! endfor
%! endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! for k=1:2
%! A = [1 0 0 0 0
%! 0 3 0 0 0
%! 2 0 5 0 0
%! 0 0 6 7 0
%! 0 0 0 0 8];
%! b = [1, 9;
%!      2, 5;
%!      3, 4;
%!      4, 8;
%!      6, 7];
%! switch (k)
%! case 2
%! A += 0.5j * A;
%! b += 0.5j * b;
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.factorization = PASTIX_API_FACT_LU;
%! opts.matrix_type = PASTIX_API_SYM_NO;
%! opts.refine_max_iter = int32(10);
%! opts.check_solution = true;
%! x = pastix(A, b, opts);
%! f = A * x - b;
%! assert(norm(f) < eps^0.8 * norm(b));
%! assert(x, A \ b, eps^0.8 * norm(A \ b));
%! endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! tol = eps^0.35;
%! rand("seed", 0);
%! for N=[10, 20, 100]
%! for i=1:10
%! for j=1:2
%! for k=1:2
%! A = rand(N, N);
%! b = rand(N, 10);
%! switch (k)
%! case 2
%! A += 1j * rand(N, N);
%! b += 1j * rand(N, 10);
%! endswitch
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.factorization = PASTIX_API_FACT_LU;
%! opts.matrix_type = PASTIX_API_SYM_NO;
%! opts.refine_max_iter = int32(10);
%! opts.check_solution = true;
%! switch j
%! case 1
%! x = pastix(A, b, opts);
%! case 2
%! x = pastix(pastix(A, opts), b);
%! endswitch
%! f = A * x - b;
%! assert(all(norm(f, "cols") < tol * norm(b, "cols")));
%! assert(x, A \ b, tol * norm(A \ b, "cols"));
%! endfor
%! endfor
%! endfor
%! endfor
%! endif

%!test
%! test_idx = int32(0);
%! if (numel(which("pastix")))
%!   rand("seed", 0);
%!   for mt=[PASTIX_API_SYM_YES, PASTIX_API_SYM_NO]
%!     for ref=[true,false]
%!       for t=[1,4]
%! 	for s=0:2
%! 	  for j=1:2
%! 	    for k=1:2
%! 	      switch mt
%! 		case PASTIX_API_SYM_NO
%! 		  frange = PASTIX_API_FACT_LU;
%! 		otherwise
%! 		  frange = [PASTIX_API_FACT_LDLT, PASTIX_API_FACT_LLT];
%! 	      endswitch
%! 	      for f=frange
%! 		for N=[10, 50]
%! 		  for i=1:10
%! 		    for j=1:2
%! 		      A = sprand(N, N, 0.1, 1) + diag(rand(N, 1) + 1);
%! 		      switch (k)
%! 			case 2
%!                        [r, c, d] = find(A);
%! 			  A += sparse(r, c, 1j * rand(numel(d), 1), N, N);
%! 		      endswitch
%! 		      switch mt
%! 			case PASTIX_API_SYM_NO
%! 			case PASTIX_API_SYM_YES
%! 			  switch f
%! 			    case {PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT}
%! 			      A *= A.';
%! 			  endswitch
%! 		      endswitch
%! 		      [r, c, d] = find(A);
%! 		      opts.factorization = f;
%! 		      switch mt
%! 			case PASTIX_API_SYM_NO
%! 			  idx = 1:numel(r);
%! 			otherwise
%! 			  switch s
%! 			    case 0
%! 			      idx = find(r >= c);
%! 			    case 1
%! 			      idx = find(r <= c);
%! 			    otherwise
%! 			      idx = 1:numel(r);
%! 			  endswitch
%! 		      endswitch
%! 		      b = rand(N, 10);
%! 		      switch (k)
%! 			case 2
%! 			  b += 1j * rand(N, 10);
%! 		      endswitch
%! 		      opts.verbose = PASTIX_API_VERBOSE_NOT;
%! 		      opts.refine_max_iter = ref;
%! 		      opts.matrix_type = mt;
%! 		      opts.number_of_threads = t;
%! 		      opts.check_solution = true;
%! 		      Asym = sparse(r(idx), c(idx), d(idx));
%! 		      assert(nnz(Asym) > 0);
%! 		      xref = A \ b;
%! 		      switch j
%! 			case 1
%! 			  x = pastix(Asym, b, opts);
%! 			case 2
%! 			  x = pastix(pastix(Asym, opts), b);
%! 		      endswitch
%! 		      if ref
%! 			tolf = eps^0.45;
%! 			tolx = eps^0.45;
%! 		      else
%! 			tolf = eps^0.3;
%! 			tolx = eps^0.4;
%! 		      endif
%! 		      fpas = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%! 		      fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%! 		      assert(max(fpas) < tolf);
%! 		      assert(max(fref) < tolf);
%! 		      assert(x, xref, tolx * max(norm(xref, "cols")));
%! 		      fprintf(stderr, "current test %d passed\n", ++test_idx);
%! 		    endfor
%! 		  endfor
%! 		endfor
%! 	      endfor
%! 	    endfor
%! 	  endfor
%! 	endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! if (numel(which("pastix")))
%! rand("seed", 0);
%! for mt=[PASTIX_API_SYM_YES]
%! for s=0:2
%! for f=[PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT]
%! for N=[2, 10, 100]
%! for i=1:10
%! for k=1:2
%! A = rand(N, N)+2 * diag(rand(N,1));
%! switch (k)
%! case 2
%! A += 1j * (rand(N, N)+2 * diag(rand(N,1)));
%! endswitch
%! switch mt
%! case PASTIX_API_SYM_NO
%! case PASTIX_API_SYM_YES
%! switch f
%! case PASTIX_API_FACT_LLT
%! A *= A.';
%! case PASTIX_API_FACT_LDLT
%! A *= A.';
%! endswitch
%! endswitch
%! [r, c, d] = find(A);
%! opts.factorization = f;
%! switch s
%! case 0
%! idx = find(r >= c);
%! case 1
%! idx = find(r <= c);
%! otherwise
%! idx = 1:numel(r);
%! endswitch
%! b = rand(N, 10);
%! opts.verbose = PASTIX_API_VERBOSE_NOT;
%! opts.refine_max_iter = int32(10);
%! opts.matrix_type = mt;
%! opts.number_of_threads = 1;
%! opts.check_solution = true;
%! Asym = sparse(r(idx), c(idx), d(idx));
%! assert(nnz(Asym) > 0);
%! x = pastix(Asym, b, opts);
%! xref = A \ b;
%! tol = eps^0.3;
%! fpas = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%! fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%! assert(max(fpas) < tol);
%! assert(max(fref) < tol);
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endfor
%! endif

%!error mumps([]);
%!error mumps(eye(3));
%!error mumps(eye(3), zeros(3,1));
%!error mumps(mumps(eye(3), struct()));
%!error mumps(eye(3),zeros(2,1),struct());
%!error mumps(ones(3,2),zeros(3,1),struct());
%!test
%! if (numel(which("mumps")))
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
%! else
%! warning("mumps is not installed");
%! endif

%!test
%! if (numel(which("ndmetis")))
%! eptr = int32([1, 10, 12]);
%! eind = int32([2, 30, 1, 40, 9, 7, 55, 80, 77, 13, 5, 100]);
%! nn = int32(100);
%! [perm, iperm] = ndmetis(nn, eptr, eind);
%! else
%! warning("ndmetis is not installed");
%! endif

%!error umfpack([]);
%!error umfpack(eye(3));
%!error umfpack(eye(3), zeros(3,1));
%!error umfpack(umfpack(eye(3), struct()));
%!error umfpack(eye(3),zeros(2,1),struct());
%!error umfpack(ones(3,2),zeros(3,1),struct());
%!test
%! if (numel(which("umfpack")))
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
%!             for m=1:2
%!             switch (l)
%!               case 1
%!                 A = rand(n(j),n(j));
%!                 switch (m)
%!                 case 2
%!                   A += 1j * rand(n(j), n(j));
%!                 endswitch
%!               otherwise
%!                 A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                 switch (m)
%!                 case 2
%!                   A = 1j * (sprand(n(j), n(j), 0.1) + diag(rand(n(j),1)));
%!                 endswitch
%!             endswitch
%!             Af = A;
%!             b = rand(n(j), 3);
%!             switch (m)
%!             case 2
%!               b += 1j * rand(n(j), 3);
%!             endswitch
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
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endfor
%! unwind_protect_cleanup
%! rand("state", state);
%! end_unwind_protect
%! else
%! warning("umfpack is not installed");
%! endif

%!test
%! if (numel(which("dsbgvx")))
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
%! else
%! warning("dsbgvx is not installed");
%! endif

%!test
%! if (numel(which("dsbgvx")))
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
%! endif

%!test
%! if (numel(which("dsbgvx")))
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
%! endif

%!test
%! if (numel(which("dsbgvx")))
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
%! endif

%!function vn = normalize_U(v)
%! idxmax = find(abs(v) == max(abs(v)))(1);
%! vn = v / v(idxmax);

%!test
%! if (numel(which("dspev")))
%! rand("seed", 0);
%! for i=1:10
%!   A = zeros(200,200);
%!   B = rand(10);
%!   for j=1:size(A,1)/10
%!     A((j-1)*10 + (1:10),(j-1)*10 + (1:10)) = rand()*B;
%!   endfor
%!   A = A + A.';
%!  [U1,lambda1]=dspev(A);
%!  [U,lambda]=eig(A);
%!  lambda = diag(lambda).';
%!  tol_lambda = sqrt(eps)*max(abs(lambda));
%!  assert(lambda1,lambda,tol_lambda);
%!  for j=1:size(U,2)
%!    assert(normalize_U(U1(:, j)), normalize_U(U(:, j)), sqrt(eps));
%!  endfor
%! endfor
%! else
%! warning("dspev is not installed");
%! endif

%!test
%! if (numel(which("dspev")))
%! rand("seed", 0);
%! format long g;
%! for i=1:10
%!  A = sparse([],[],[],200,200);
%!  B = rand(10);
%!  for j=1:size(A,1)/10
%!     A((j-1)*10 + (1:10),(j-1)*10 + (1:10)) = rand()*B;
%!  endfor
%!  A = A + A.';
%!  [U1, lambda1] = dspev(A);
%!  [U,lambda]=eig(A);
%!  lambda = diag(lambda).';
%!  tol_lambda = sqrt(eps) * max(abs(lambda));
%!  assert(lambda1,lambda,tol_lambda);
%!  for j=1:size(U,2)
%!    assert(normalize_U(U1(:, j)), normalize_U(U(:, j)), sqrt(eps));
%!  endfor
%! endfor
%! endif

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
