## Copyright (C) 2011(-2023) Reinhard <octave-user@a1.net>
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
## This package contains interfaces to several well known numerical solvers like pastix, mumps, umfpack, pardiso, lapack, arpack and metis.

%!test
%! if (~isempty(which("pastix")))
%!   for i=1:2
%!     for j=1:2
%!       A = [1 0 0 0 0
%!            0 3 0 0 0
%!            2 0 5 0 0
%!            0 0 6 7 0
%!            0 0 0 0 8];
%!       b = [1, 9;
%!            2, 5;
%!            3, 4;
%!            4, 8;
%!            6, 7];
%!       switch (j)
%!         case 2
%!           A += 0.5j * A;
%!           b += 0.5j * b;
%!       endswitch
%!       opts.verbose = PASTIX_API_VERBOSE_NOT;
%!       opts.factorization = PASTIX_API_FACT_LU;
%!       opts.matrix_type = PASTIX_API_SYM_NO;
%!       opts.refine_max_iter = int32(10);
%!       opts.bind_thread_mode = PASTIX_API_BIND_NO;
%!       opts.number_of_threads = int32(4);
%!       opts.check_solution = true;
%!       switch i
%!         case 1
%!           x = pastix(A, b, opts);
%!         case 2
%!           x = pastix(pastix(A, opts), b);
%!       endswitch
%!       f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!       assert(f <= eps^0.8);
%!     endfor
%!   endfor
%! else
%!   warning("pastix is not installed");
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   tol = eps^0.35;
%!   rand("seed", 0);
%!   for N=[2, 10, 100]
%!     for i=1:10
%!       for j=1:2
%!         for k=1:2
%!           A = rand(N, N);
%!           b = rand(N, 10);
%!           switch (k)
%!             case 2
%!               A += 1j * rand(N, N);
%!               b += 1j * rand(N, 10);
%!           endswitch
%!           opts.verbose = PASTIX_API_VERBOSE_NOT;
%!           opts.factorization = PASTIX_API_FACT_LU;
%!           opts.matrix_type = PASTIX_API_SYM_NO;
%!           opts.refine_max_iter = int32(10);
%!           opts.check_solution = true;
%!           switch j
%!             case 1
%!               x = pastix(A, b, opts);
%!             case 2
%!               x = pastix(pastix(A, opts), b);
%!           endswitch
%!           f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!           assert(f <= tol);
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   tol = eps^0.35;
%!   rand("seed", 0);
%!   for N=[2, 10, 100]
%!     for i=1:10
%!       for j=1:2
%!         for k=1:2
%!           A = rand(N, N);
%!           b = rand(N, 10);
%!           switch (k)
%!             case 2
%!               A += 1j * rand(N, N);
%!               b += 1j * rand(N, 10);
%!           endswitch
%!           opts.verbose = PASTIX_API_VERBOSE_NOT;
%!           opts.factorization = PASTIX_API_FACT_LU;
%!           opts.matrix_type = PASTIX_API_SYM_NO;
%!           opts.refine_max_iter = int32(10);
%!           opts.check_solution = true;
%!           switch j
%!             case 1
%!               x = pastix(A, b, opts);
%!             case 2
%!               x = pastix(pastix(A, opts), b);
%!           endswitch
%!           f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!           assert(f <= tol);
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   tol = eps^0.3;
%!   rand("seed", 0);
%!   for ref=int32([20])
%!     for bind=[PASTIX_API_BIND_NO]
%!       for t=[1,4]
%!         for s=0:1
%!           for f=[PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT]
%!             for N=[10, 100]
%!               for i=1:10
%!                 for j=1:2
%!                   for k=1:2
%!                     A = rand(N, N);
%!                     switch (k)
%!                       case 2
%!                         A += 1j * rand(N, N);
%!                     endswitch
%!                     A *= A.';
%!                     [r, c, d] = find(A);
%!                     if s
%!                       idx = find(r >= c);
%!                     else
%!                       idx = 1:numel(r);
%!                     endif
%!                     b = rand(N, 10);
%!                     switch (k)
%!                       case 2
%!                         b += 1j * rand(N, 10);
%!                     endswitch
%!                     opts.verbose = PASTIX_API_VERBOSE_NOT;
%!                     opts.refine_max_iter = ref;
%!                     opts.matrix_type = PASTIX_API_SYM_YES;
%!                     opts.factorization = f;
%!                     opts.number_of_threads = t;
%!                     opts.bind_thread_mode = bind;
%!                     opts.check_solution = false;
%!                     opts.epsilon_refinement = 1e-12;
%!                     switch j
%!                       case 1
%!                         x = pastix(sparse(r(idx), c(idx), d(idx)), b, opts);
%!                       case 2
%!                         x = pastix(pastix(sparse(r(idx), c(idx), d(idx)), opts), b);
%!                     endswitch
%!                     xref = A \ b;
%!                     ferr = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%!                     fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%!                     assert(max(ferr) <= tol);
%!                     assert(max(ferr) <= 10 * max(fref));
%!                   endfor
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   for k=1:2
%!     tol = sqrt(eps);
%!     A = [ 1, -1,  0, 0,  1;
%!           -1,  2, -1, 0,  0;
%!           0, -1,  2, -1, 0;
%!           0,  0, -1,  1, 0;
%!           1,  0,  0,  0, 0];

%!     b = [1; 2; 3; 4; 5];
%!     switch (k)
%!       case 2
%!         A += 0.5j * A;
%!         b += 0.5j * b;
%!     endswitch
%!     opts.verbose = PASTIX_API_VERBOSE_NOT;
%!     opts.refine_max_iter = int32(10);
%!     opts.matrix_type = PASTIX_API_SYM_YES;
%!     opts.factorization = PASTIX_API_FACT_LDLT;
%!     opts.check_solution = true;
%!     for i=1:2
%!       [r, c, d] = find(A);
%!       idx = find(r >= c);
%!       Asym = sparse(r(idx), c(idx), d(idx), rows(A), columns(A));
%!       switch i
%!         case 1
%!           x = pastix(Asym, b, opts);
%!         case 2
%!           x = pastix(pastix(Asym, opts), b);
%!       endswitch
%!       f = max(norm(A * x - b) ./ norm(A * x + b));
%!       assert(f <= tol);
%!       assert(max(norm(x - A \ b, "cols")) <= tol * max(norm(A \ b, "cols")));
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   for k=1:2
%!     A = [1 0 0 0 0
%!          0 3 0 0 0
%!          2 0 5 0 0
%!          0 0 6 7 0
%!          0 0 0 0 8];
%!     b = [1, 9;
%!          2, 5;
%!          3, 4;
%!          4, 8;
%!          6, 7];
%!     switch (k)
%!       case 2
%!         A += 0.5j * A;
%!         b += 0.5j * b;
%!     endswitch
%!     opts.verbose = PASTIX_API_VERBOSE_NOT;
%!     opts.factorization = PASTIX_API_FACT_LU;
%!     opts.matrix_type = PASTIX_API_SYM_NO;
%!     opts.refine_max_iter = int32(10);
%!     opts.check_solution = true;
%!     x = pastix(A, b, opts);
%!     f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!     assert(f <= eps^0.8);
%!     assert(max(norm(x - A \ b, "cols")) <= eps^0.8 * max(norm(A \ b, "cols")));
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   tol = eps^0.35;
%!   rand("seed", 0);
%!   for N=[10, 20, 100]
%!     for i=1:10
%!       for j=1:2
%!         for k=1:2
%!           A = rand(N, N);
%!           b = rand(N, 10);
%!           switch (k)
%!             case 2
%!               A += 1j * rand(N, N);
%!               b += 1j * rand(N, 10);
%!           endswitch
%!           opts.verbose = PASTIX_API_VERBOSE_NOT;
%!           opts.factorization = PASTIX_API_FACT_LU;
%!           opts.matrix_type = PASTIX_API_SYM_NO;
%!           opts.refine_max_iter = int32(10);
%!           opts.check_solution = true;
%!           switch j
%!             case 1
%!               x = pastix(A, b, opts);
%!             case 2
%!               x = pastix(pastix(A, opts), b);
%!           endswitch
%!           f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!           assert(f <= tol);
%!           assert(max(norm(x - A \ b, "cols")) <= tol * max(norm(A \ b, "cols")));
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! test_idx = int32(0);
%! if (~isempty(which("pastix")))
%!   rand("seed", 0);
%!   for mt=[PASTIX_API_SYM_YES, PASTIX_API_SYM_NO]
%!     for ref=[true,false]
%!       for t=[1,4]
%!         for s=0:2
%!           for j=1:2
%!             for k=1:2
%!               switch mt
%!                 case PASTIX_API_SYM_NO
%!                   frange = PASTIX_API_FACT_LU;
%!                 otherwise
%!                   frange = [PASTIX_API_FACT_LDLT, PASTIX_API_FACT_LLT];
%!               endswitch
%!               for f=frange
%!                 for N=[10, 50]
%!                   for i=1:10
%!                     for j=1:2
%!                       A = sprand(N, N, 0.1, 1) + diag(rand(N, 1) + 1);
%!                       switch (k)
%!                         case 2
%!                           [r, c, d] = find(A);
%!                           A += sparse(r, c, 1j * rand(numel(d), 1), N, N);
%!                       endswitch
%!                       switch mt
%!                         case PASTIX_API_SYM_NO
%!                         case PASTIX_API_SYM_YES
%!                           switch f
%!                             case {PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT}
%!                               A *= A.';
%!                           endswitch
%!                       endswitch
%!                       [r, c, d] = find(A);
%!                       opts.factorization = f;
%!                       switch mt
%!                         case PASTIX_API_SYM_NO
%!                           idx = 1:numel(r);
%!                         otherwise
%!                           switch s
%!                             case 0
%!                               idx = find(r >= c);
%!                             case 1
%!                               idx = find(r <= c);
%!                             otherwise
%!                               idx = 1:numel(r);
%!                           endswitch
%!                       endswitch
%!                       b = rand(N, 10);
%!                       switch (k)
%!                         case 2
%!                           b += 1j * rand(N, 10);
%!                       endswitch
%!                       opts.verbose = PASTIX_API_VERBOSE_NOT;
%!                       opts.refine_max_iter = ref;
%!                       opts.matrix_type = mt;
%!                       opts.number_of_threads = t;
%!                       opts.check_solution = true;
%!                       Asym = sparse(r(idx), c(idx), d(idx));
%!                       assert(nnz(Asym) > 0);
%!                       xref = A \ b;
%!                       switch j
%!                         case 1
%!                           x = pastix(Asym, b, opts);
%!                         case 2
%!                           x = pastix(pastix(Asym, opts), b);
%!                       endswitch
%!                       if ref
%!                         tolf = eps^0.45;
%!                         tolx = eps^0.45;
%!                       else
%!                         tolf = eps^0.3;
%!                         tolx = eps^0.4;
%!                       endif
%!                       fpas = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%!                       fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%!                       assert(max(fpas) <= tolf);
%!                       assert(max(fref) <= tolf);
%!                       assert(max(norm(x - xref, "cols")) <= tolx * max(norm(xref, "cols")));
%!                       fprintf(stdout, "current test %d passed\n", ++test_idx);
%!                     endfor
%!                   endfor
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("pastix")))
%!   rand("seed", 0);
%!   for mt=[PASTIX_API_SYM_YES]
%!     for s=0:2
%!       for f=[PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT]
%!         for N=[2, 10, 100]
%!           for i=1:10
%!             for k=1:2
%!               A = rand(N, N)+2 * diag(rand(N,1));
%!               switch (k)
%!                 case 2
%!                   A += 1j * (rand(N, N)+2 * diag(rand(N,1)));
%!               endswitch
%!               switch mt
%!                 case PASTIX_API_SYM_NO
%!                 case PASTIX_API_SYM_YES
%!                   switch f
%!                     case PASTIX_API_FACT_LLT
%!                       A *= A.';
%!                     case PASTIX_API_FACT_LDLT
%!                       A *= A.';
%!                   endswitch
%!               endswitch
%!               [r, c, d] = find(A);
%!               opts.factorization = f;
%!               switch s
%!                 case 0
%!                   idx = find(r >= c);
%!                 case 1
%!                   idx = find(r <= c);
%!                 otherwise
%!                   idx = 1:numel(r);
%!               endswitch
%!               b = rand(N, 10);
%!               opts.verbose = PASTIX_API_VERBOSE_NOT;
%!               opts.refine_max_iter = int32(10);
%!               opts.matrix_type = mt;
%!               opts.number_of_threads = 1;
%!               opts.check_solution = false;
%!               opts.epsilon_refinement = eps^0.9;
%!               Asym = sparse(r(idx), c(idx), d(idx));
%!               assert(nnz(Asym) > 0);
%!               x = pastix(Asym, b, opts);
%!               xref = A \ b;
%!               tol = eps^0.3;
%!               fpas = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%!               fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%!               assert(max(fpas) <= tol);
%!               assert(max(fref) <= tol);
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif

%!error mumps([]);
%!error mumps(eye(3));
%!error mumps(eye(3), zeros(3,1));
%!error mumps(mumps(eye(3), struct()));
%!error mumps(eye(3),zeros(2,1),struct());
%!error mumps(ones(3,2),zeros(3,1),struct());
%!test
%! if (~isempty(which("mumps")))
%!   rand("seed", 0);
%!   n = [2,4,8,16,32,64,128];
%!   for compl=[false, true]
%!     for e=[0,100]
%!       for u=1:2
%!         for m=[MUMPS_MAT_GEN, MUMPS_MAT_DEF, MUMPS_MAT_SYM]
%!           for i=1:10
%!             for j=1:numel(n)
%!               for k=1:2
%!                 for l=1:2
%!                   for s=1:2
%!                     switch (l)
%!                       case 1
%!                         A = rand(n(j),n(j));
%!                       otherwise
%!                         A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                     endswitch
%!                     if (compl)
%!                       A *= (rand() + 1j * rand());
%!                     endif
%!                     Af = A;
%!                     b = rand(n(j), 3);
%!                     switch (s)
%!                       case {1,2}
%!                         switch (s)
%!                           case 2
%!                             A += A.';
%!                           case 1
%!                             A *= A.';
%!                             A += eye(size(A));
%!                         endswitch
%!                         Af = A;
%!                         if (s == 2 && m > 0)
%!                           [r, c, d] = find(A);
%!                           if (u == 1)
%!                             idx = find(r >= c);
%!                           else
%!                             idx = find(r <= c);
%!                           endif
%!                           r = r(idx);
%!                           c = c(idx);
%!                           d = d(idx);
%!                           Af = sparse(r, c, d, n(j), n(j));
%!                           if (l == 1)
%!                             Af = full(Af);
%!                           endif
%!                         endif
%!                     endswitch
%!                     xref = A \ b;
%!                     opt.verbose = MUMPS_VER_WARN;
%!                     opt.refine_max_iter = e;
%!                     opt.matrix_type = m;
%!                     switch (k)
%!                       case 1
%!                         x = mumps(Af, b, opt);
%!                       otherwise
%!                         x = mumps(mumps(Af, opt), b);
%!                     endswitch
%!                     assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                     assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(A*x+b));
%!                     assert(max(norm(x - xref, "cols")) <= sqrt(eps) * norm(x));
%!                   endfor
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! else
%!   warning("mumps is not installed");
%! endif

%!test
%! if (~isempty(which("ndmetis")))
%!   eptr = int32([1, 10, 12]);
%!   eind = int32([2, 30, 1, 40, 9, 7, 55, 80, 77, 13, 5, 100]);
%!   nn = int32(100);
%!   [perm, iperm] = ndmetis(nn, eptr, eind);
%! else
%!   warning("ndmetis is not installed");
%! endif

%!error umfpack([]);
%!error umfpack(eye(3));
%!error umfpack(eye(3), zeros(3,1));
%!error umfpack(umfpack(eye(3), struct()));
%!error umfpack(eye(3),zeros(2,1),struct());
%!error umfpack(ones(3,2),zeros(3,1),struct());
%!test
%! if (~isempty(which("umfpack")))
%!   state = rand("state");
%!   unwind_protect
%!     rand("seed", 0);
%!     n = [2,4,8,16,32,64,128];
%!     for e=[0,100]
%!       for u=1:2
%!         for i=1:10
%!           for j=1:numel(n)
%!             for k=1:2
%!               for l=1:2
%!                 for m=1:2
%!                   switch (l)
%!                     case 1
%!                       A = rand(n(j),n(j));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * rand(n(j), n(j));
%!                       endswitch
%!                     otherwise
%!                       A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * (sprand(n(j), n(j), 0.1) + diag(rand(n(j),1)));
%!                       endswitch
%!                   endswitch
%!                   Af = A;
%!                   b = rand(n(j), 3);
%!                   switch (m)
%!                     case 2
%!                       b += 1j * rand(n(j), 3);
%!                   endswitch
%!                   xref = A \ b;
%!                   opt.verbose = 0;
%!                   opt.refine_max_iter = e;
%!                   switch (k)
%!                     case 1
%!                       x = umfpack(A, b, opt);
%!                     otherwise
%!                       x = umfpack(umfpack(A, opt), b);
%!                   endswitch
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(A*x+b));
%!                   assert(max(norm(x - xref, "cols")) <= sqrt(eps) * norm(x));
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   unwind_protect_cleanup
%!     rand("state", state);
%!   end_unwind_protect
%! else
%!   warning("umfpack is not installed");
%! endif

%!error strumpack([]);
%!error strumpack(eye(3));
%!error strumpack(eye(3), zeros(3,1));
%!error strumpack(strumpack(eye(3), struct()));
%!error strumpack(eye(3),zeros(2,1),struct());
%!error strumpack(ones(3,2),zeros(3,1),struct());
%!test
%! if (~isempty(which("strumpack")))
%!   state = rand("state");
%!   unwind_protect
%!     rand("seed", 0);
%!     n = [2,4,8,16,32,64,128];
%!     for e=[0,100]
%!       for u=1:2
%!         for i=1:10
%!           for j=1:numel(n)
%!             for k=1:2
%!               for l=1:2
%!                 for m=1:2
%!                   switch (l)
%!                     case 1
%!                       A = rand(n(j),n(j));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * rand(n(j), n(j));
%!                       endswitch
%!                     otherwise
%!                       A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * (sprand(n(j), n(j), 0.1) + diag(rand(n(j),1)));
%!                       endswitch
%!                   endswitch
%!                   Af = A;
%!                   b = rand(n(j), 3);
%!                   switch (m)
%!                     case 2
%!                       b += 1j * rand(n(j), 3);
%!                   endswitch
%!                   xref = A \ b;
%!                   opt.verbose = 0;
%!                   opt.refine_max_iter = e;
%!                   switch (k)
%!                     case 1
%!                       x = strumpack(A, b, opt);
%!                     otherwise
%!                       x = strumpack(strumpack(A, opt), b);
%!                   endswitch
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * max(norm(A*x+b, "cols")));
%!                   assert(max(norm(x - xref, "cols")) <= sqrt(eps) * norm(x));
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   unwind_protect_cleanup
%!     rand("state", state);
%!   end_unwind_protect
%! else
%!   warning("strumpack is not installed");
%! endif

%!test
%! if (~isempty(which("dsbgvx")))
%!   clear all;
%!   rand("seed", 0);
%!   L = 0.5;
%!   U = 0.9;
%!   RANGE = 'V';
%!   for j=1:10
%!     A = sprand(10,10,0.3);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(10,10,0.3);
%!     B = R.' * R + eye(10);
%!     P = symrcm(B);
%!     B = B(P,P);

%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));

%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(max(norm(A*x(:,i) - lambda(i)*B*x(:,i), "cols")) <= sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! else
%!   warning("dsbgvx is not installed");
%! endif

%!test
%! if (~isempty(which("dsbgvx")))
%!   rand("seed", 0);
%!   L = 1;
%!   U = 3;
%!   RANGE = 'I';
%!   for j=1:10
%!     A = sprand(10,10,0.3);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(10,10,0.3);
%!     B = R.' * R + eye(10);
%!     P = symrcm(B);
%!     B = B(P,P);

%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));

%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(norm(A*x(:,i) - lambda(i)*B*x(:,i)) <= sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("dsbgvx")))
%!   rand("seed", 0);
%!   L = 1;
%!   U = 10;
%!   RANGE = 'A';
%!   for j=1:100
%!     A = sprand(10,10,0.3) + eye(10);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(10,10,0.3) + eye(10);
%!     B = R.' * R;
%!     P = symrcm(B);
%!     B = B(P,P);

%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));

%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(norm(A*x(:,i) - lambda(i)*B*x(:,i)) < sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! endif

%!test
%! if (~isempty(which("dsbgvx")))
%!   rand("seed", 0);
%!   L = 1;
%!   U = 3;
%!   N = 1000;
%!   RANGE = 'I';
%!   for j=1:3
%!     A = sprand(N,N,0.001) + eye(N);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(N,N,0.001) + eye(N);
%!     B = R.' * R;
%!     P = symrcm(B);
%!     B = B(P,P);

%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));

%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(norm(A*x(:,i) - lambda(i)*B*x(:,i)) < eps^0.4*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! endif

%!function vn = normalize_U(v)
%!   idxmax = find(abs(v) == max(abs(v)))(1);
%!   vn = v / v(idxmax);

%!test
%!   if (~isempty(which("dspev")))
%!     rand("seed", 0);
%!     for i=1:10
%!       A = zeros(200,200);
%!       B = rand(10);
%!       for j=1:size(A,1)/10
%!         A((j-1)*10 + (1:10),(j-1)*10 + (1:10)) = rand()*B;
%!       endfor
%!       A = A + A.';
%!       [U1,lambda1]=dspev(A);
%!       [U,lambda]=eig(A);
%!       lambda = diag(lambda).';
%!       tol_lambda = sqrt(eps)*max(abs(lambda));
%!       assert(norm(lambda1 - lambda) <= tol_lambda);
%!       for j=1:size(U,2)
%!         assert(norm(normalize_U(U1(:, j)) - normalize_U(U(:, j))) <= sqrt(eps));
%!       endfor
%!     endfor
%!   else
%!     warning("dspev is not installed");
%!   endif

%!test
%!   if (~isempty(which("dspev")))
%!     rand("seed", 0);
%!     format long g;
%!     for i=1:10
%!       A = sparse([],[],[],200,200);
%!       B = rand(10);
%!       for j=1:size(A,1)/10
%!         A((j-1)*10 + (1:10),(j-1)*10 + (1:10)) = rand()*B;
%!       endfor
%!       A = A + A.';
%!       [U1, lambda1] = dspev(A);
%!       [U,lambda]=eig(A);
%!       lambda = diag(lambda).';
%!       tol_lambda = sqrt(eps) * max(abs(lambda));
%!       assert(norm(lambda1 - lambda) <= tol_lambda);
%!       for j=1:size(U,2)
%!         assert(norm(normalize_U(U1(:, j)) - normalize_U(U(:, j))) <= sqrt(eps));
%!       endfor
%!     endfor
%!   endif

%!test
%!  if (isempty(which("sp_sym_mtimes")))
%!    warning("sp_sym_mtimes was not installed");
%!    return
%!  endif
%!   state = rand("state");
%!   unwind_protect
%!     rand("seed", 0);
%!     tol = eps^0.9;
%!     for i=1:10
%!       for N=[1, 10, 100, 1000]
%!         for M=[1, 10, 20, 100]
%!           A = sprand(N, N, 0.01);
%!           A += A.';
%!           x = rand(N, M);
%!           b = A * x;
%!           [r, c, d] = find(A);
%!           for k=1:2
%!             switch (k)
%!               case 1
%!                 idx = find(r >= c);
%!               case 2
%!                 idx = find(r <= c);
%!             endswitch
%!             A2 = sparse(r(idx), c(idx), d(idx), N, N);
%!             b2 = sp_sym_mtimes(A2, x);
%!             assert(norm(b2 - b) <= tol * norm(b));
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   unwind_protect_cleanup
%!     rand("state", state);
%!   end_unwind_protect

%!test
%!   test_idx = int32(0);
%!   if (~isempty(which("pardiso")))
%!     rand("seed", 0);
%!     for z=[true, false]
%!       for o=[0,2,3]
%!         for sca=[0,1]
%!           for wm=[1]
%!             for ooc=[0]
%!               for sym=[true, false]
%!                 for ref=[true,false]
%!                   for t=[1,4]
%!                     for s=0:4
%!                       for j=1:2
%!                         for N=[10, 50]
%!                           for i=1
%!                             for j=1:2
%!                               switch (s)
%!                                 case 3
%!                                   A = sprand(N, N, 0.9, 1);
%!                                   if (z)
%!                                     A += 1j * sprand(N, N, 0.9, 1);
%!                                   endif
%!                                   A -= diag(diag(A));
%!                                 case 4
%!                                   A = sprand(N, N, 0.9, 1);
%!                                   if (z)
%!                                     A += 1j * sprand(N, N, 0.9, 1);
%!                                   endif
%!                                   for k=1:N
%!                                     if (rand() > 0.5)
%!                                       A(k, k) = 0;
%!                                     endif
%!                                   endfor
%!                                 otherwise
%!                                   A = sprand(N, N, 0.1, 1) + 40 * diag(rand(N, 1));
%!                               endswitch
%!                               if (sym)
%!                                 A += A.';
%!                               endif
%!                               if (rank(A) < N)
%!                                 continue;
%!                               endif
%!                               [r, c, d] = find(A);
%!                               switch(sym)
%!                                 case false
%!                                   idx = 1:numel(r);
%!                                 otherwise
%!                                   switch s
%!                                     case 0
%!                                       idx = find(r >= c);
%!                                     case 1
%!                                       idx = find(r <= c);
%!                                     otherwise
%!                                       idx = 1:numel(r);
%!                                   endswitch
%!                               endswitch
%!                               b = rand(N, 10);
%!                               opts.symmetric = sym;
%!                               opts.ordering = o;
%!                               opts.verbose = 0;
%!                               opts.refine_max_iter = ref * 100;
%!                               opts.number_of_threads = t;
%!                               opts.out_of_core_mode = ooc;
%!                               opts.scaling = sca;
%!                               opts.weighted_matching = wm;
%!                               Asym = sparse(r(idx), c(idx), d(idx), size(A));
%!                               assert(nnz(Asym) > 0);
%!                               xref = A \ b;
%!                               switch (j)
%!                                 case 1
%!                                   x = pardiso(Asym, b, opts);
%!                                 case 2
%!                                   x = pardiso(pardiso(Asym, opts), b);
%!                               endswitch
%!                               tolf = eps^0.5;
%!                               tolx = eps^0.5;
%!                               fpar = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%!                               fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%!                               assert(max(fpar) <= tolf);
%!                               assert(max(fref) <= tolf);
%!                               assert(max(norm(x - xref, "cols")) <= tolx * max(norm(xref, "cols")));
%!                               fprintf(stdout, "current test %d passed\n", ++test_idx);
%!                             endfor
%!                           endfor
%!                         endfor
%!                       endfor
%!                     endfor
%!                   endfor
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   else
%!     warning("pardiso is not available");
%!   endif

%!function [A,B]=build_test_mat(N)
%! B = gallery("poisson", N);
%! A = gallery("tridiag", columns(B));

%!test
%! if (isempty(which("eig_sym")))
%!   warning("eig_sym was not installed");
%!   return;
%! endif
%! rand("seed", 0);
%! for k=[5, 7, 10]
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
%!  assert(norm(v1 - v2) <= tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! if (isempty(which("eig_sym")))
%!   warning("eig_sym was not installed");
%!   return;
%! endif
%! rand("seed", 0);
%! for l=[5, 7, 10]
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
%!  assert(norm(v1 - v2) <= tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! if (isempty(which("eig_sym")))
%!   warning("eig_sym was not installed");
%!   return;
%! endif
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
%! assert(norm(v1 - v) <= eps * norm(v));
%! assert(norm(lambda1 - lambda) <= eps * norm(lambda));
%! endif
%! endfor
%! tol = sqrt(eps);
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(norm(v1 - v2) <= tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! if (isempty(which("eig_sym")))
%!   warning("eig_sym was not installed");
%!   return;
%! endif
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
%!   assert(norm(lambda - lambda1) <= eps * norm(lambda));
%!   assert(norm(v - v1) <= eps * norm(v));
%! endif
%! endfor
%! tol = sqrt(eps);
%! assert(columns(lambda), nev)
%! for i=1:columns(v)
%!  v1 = A * v(:, i);
%!  v2 = lambda(i,i) * B * v(:,i);
%!  assert(norm(v1 - v2) <= tol * max([norm(v1),norm(v2)]));
%! endfor
%! endfor
%! endfor

%!test
%! if (~isempty(which("ndmetis")))
%!   eptr = int32([1, 10, 12]);
%!   eind = int32([2, 30, 1, 40, 9, 7, 55, 80, 77, 13, 5, 100]);
%!   nn = int32(100);
%!   [perm, iperm] = ndmetis(nn, eptr, eind);
%! else
%!   warning("ndmetis was not installed");
%! endif
