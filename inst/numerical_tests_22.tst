## numerical_tests.tst:22
%!function [A,B]=build_test_mat(N)
%! B = gallery("poisson", N);
%! A = gallery("tridiag", columns(B));
%!test
%! try
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
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
