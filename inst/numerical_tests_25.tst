## numerical_tests.tst:25
%!test
%! try
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
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
