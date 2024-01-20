## numerical_tests.tst:18
%!function vn = normalize_U(v)
%!   idxmax = find(abs(v) == max(abs(v)))(1);
%!   vn = v / v(idxmax);
%!test
%!   if (~isempty(which("dspev")))
%!     rand("seed", 0);
%!     for i=1:10
%!       A = zeros(200,200);
%!       B = rand(10);
%!       for j=1:size(A,1)/10
%!         A((j-1)*10 + (1:10),(j-1)*10 + (1:10)) = rand()*B;
%!       endfor
%!       A = A + A.';
%!       [U1,lambda1]=dspev(A);
%!       [U,lambda]=eig(A);
%!       lambda = diag(lambda).';
%!       tol_lambda = sqrt(eps)*max(abs(lambda));
%!       assert(norm(lambda1 - lambda) <= tol_lambda);
%!       for j=1:size(U,2)
%!         assert(norm(normalize_U(U1(:, j)) - normalize_U(U(:, j))) <= sqrt(eps));
%!       endfor
%!     endfor
%!   else
%!     warning("dspev is not installed");
%!   endif
