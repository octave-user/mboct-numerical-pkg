## numerical_tests.tst:20
%!test
%! try
%! if (isempty(which("sp_sym_mtimes")))
%!   warning("sp_sym_mtimes was not installed");
%!   return
%! endif
%! state = rand("state");
%! verbose = false;
%! unwind_protect
%!   rand("seed", 0);
%!   tol = eps^0.9;
%!   for m=1:2
%!     for l=1:2
%!       for i=1:10
%!         for N=[1, 10, 100, 1000]
%!           for M=[1, 10, 20, 100]
%!             A = sprand(N, N, 0.01) + sparse(diag(rand(N, 1) + 1.1));
%!             switch (m)
%!               case 2
%!                 A += 1j * (sprand(N, N, 0.01) + sparse(diag(rand(N, 1) + 1.1)));
%!                 assert(iscomplex(A));
%!             endswitch
%!             A += A.';
%!             x = rand(N, M);
%!             switch (l)
%!               case 2
%!                 x += 1j * rand(N, M);
%!                 assert(iscomplex(x));
%!             endswitch
%!             b = A * x;
%!             [r, c, d] = find(A);
%!             for k=1:2
%!               switch (k)
%!                 case 1
%!                   idx = find(r >= c);
%!                 case 2
%!                   idx = find(r <= c);
%!               endswitch
%!               A2 = sparse(r(idx), c(idx), d(idx), N, N);
%!               b2 = sp_sym_mtimes(A2, x);
%!               if (verbose)
%!                 tname = {"real", "complex"};
%!                 printf("%s = %s * %s\n", tname{iscomplex(b2) + 1}, tname{iscomplex(A2) + 1}, tname{iscomplex(x) + 1});
%!               endif
%!               assert(norm(b2 - b) <= tol * norm(b));
%!               if (iscomplex(A) || iscomplex(x))
%!                 assert(iscomplex(b));
%!                 assert(iscomplex(b2));
%!               else
%!                 assert(isreal(A));
%!                 assert(isreal(A2));
%!                 assert(isreal(x));
%!                 assert(isreal(b));
%!                 assert(isreal(b2));
%!               endif
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! unwind_protect_cleanup
%!   rand("state", state);
%! end_unwind_protect
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
