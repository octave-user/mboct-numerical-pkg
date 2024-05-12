## numerical_tests.tst:17
%!test
%! try
%! if (~isempty(which("dsbgvx")))
%!   rand("seed", 0);
%!   L = 1;
%!   U = 3;
%!   N = 1000;
%!   RANGE = 'I';
%!   for j=1:3
%!     A = sprand(N,N,0.001) + eye(N);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(N,N,0.001) + eye(N);
%!     B = R.' * R;
%!     P = symrcm(B);
%!     B = B(P,P);
%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));
%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(norm(A*x(:,i) - lambda(i)*B*x(:,i)) < eps^0.4*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! endif
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
