## numerical_tests.tst:21
%!test
%! try
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
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
