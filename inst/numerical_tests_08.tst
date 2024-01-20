## numerical_tests.tst:08
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
