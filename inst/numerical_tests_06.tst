## numerical_tests.tst:06
%!test
%! try
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
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
