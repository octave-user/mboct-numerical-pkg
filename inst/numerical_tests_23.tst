## numerical_tests.tst:23
%!function [A,B]=build_test_mat(N)
%! B = gallery("poisson", N);
%! A = gallery("tridiag", columns(B));
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
