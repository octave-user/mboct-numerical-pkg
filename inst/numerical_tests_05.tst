## numerical_tests.tst:05
%!test
%! try
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
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
