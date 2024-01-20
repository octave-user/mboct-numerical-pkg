## numerical_tests.tst:14
%!test
%! if (~isempty(which("dsbgvx")))
%!   clear all;
%!   rand("seed", 0);
%!   L = 0.5;
%!   U = 0.9;
%!   RANGE = 'V';
%!   for j=1:10
%!     A = sprand(10,10,0.3);
%!     A = A + A.';
%!     P = symrcm(A);
%!     A = A(P,P);
%!     R = sprand(10,10,0.3);
%!     B = R.' * R + eye(10);
%!     P = symrcm(B);
%!     B = B(P,P);
%!     assert(issymmetric(A,0));
%!     assert(isdefinite(B,0));
%!     [x,lambda] = dsbgvx(A,B,L,U,RANGE);
%!     assert(length(lambda),columns(x));
%!     assert(rows(x),rows(B));
%!     for i=1:length(lambda)
%!       assert(max(norm(A*x(:,i) - lambda(i)*B*x(:,i), "cols")) <= sqrt(eps)*norm(lambda(i)*B*x(:,i)));
%!     endfor
%!   endfor
%! else
%!   warning("dsbgvx is not installed");
%! endif
