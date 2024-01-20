## numerical_tests.tst:20
%!test
%!  if (isempty(which("sp_sym_mtimes")))
%!    warning("sp_sym_mtimes was not installed");
%!    return
%!  endif
%!   state = rand("state");
%!   unwind_protect
%!     rand("seed", 0);
%!     tol = eps^0.9;
%!     for i=1:10
%!       for N=[1, 10, 100, 1000]
%!         for M=[1, 10, 20, 100]
%!           A = sprand(N, N, 0.01);
%!           A += A.';
%!           x = rand(N, M);
%!           b = A * x;
%!           [r, c, d] = find(A);
%!           for k=1:2
%!             switch (k)
%!               case 1
%!                 idx = find(r >= c);
%!               case 2
%!                 idx = find(r <= c);
%!             endswitch
%!             A2 = sparse(r(idx), c(idx), d(idx), N, N);
%!             b2 = sp_sym_mtimes(A2, x);
%!             assert(norm(b2 - b) <= tol * norm(b));
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   unwind_protect_cleanup
%!     rand("state", state);
%!   end_unwind_protect
