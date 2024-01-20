## numerical_tests.tst:04
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
