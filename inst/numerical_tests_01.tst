## numerical_tests.tst:01
%!test
%! try
%! if (~isempty(which("pastix")))
%!   for i=1:4
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
%!       case {1, 2}
%!         trans = 0;
%!       case {3, 4}
%!         trans = 2;
%!       endswitch
%!       switch i
%!         case {1, 3}
%!           x = pastix(A, b, opts, trans);
%!         case {2, 4}
%!           x = pastix(pastix(A, opts), b, trans);
%!       endswitch
%!       switch (trans)
%!       case 0
%!         f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!       case 2
%!         f = max(norm(A.' * x - b, "cols") ./ norm(A.' * x + b, "cols"));
%!       endswitch
%!       assert(f <= eps^0.8);
%!     endfor
%!   endfor
%! else
%!   warning("pastix is not installed");
%! endif
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
