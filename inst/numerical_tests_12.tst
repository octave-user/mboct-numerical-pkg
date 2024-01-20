## numerical_tests.tst:12
%!test
%! if (~isempty(which("umfpack")))
%!   state = rand("state");
%!   unwind_protect
%!     rand("seed", 0);
%!     n = [2,4,8,16,32,64,128];
%!     for e=[0,100]
%!       for u=1:2
%!         for i=1:10
%!           for j=1:numel(n)
%!             for k=1:2
%!               for l=1:2
%!                 for m=1:2
%!                   switch (l)
%!                     case 1
%!                       A = rand(n(j),n(j));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * rand(n(j), n(j));
%!                       endswitch
%!                     otherwise
%!                       A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                       switch (m)
%!                         case 2
%!                           A += 1j * (sprand(n(j), n(j), 0.1) + diag(rand(n(j),1)));
%!                       endswitch
%!                   endswitch
%!                   Af = A;
%!                   b = rand(n(j), 3);
%!                   switch (m)
%!                     case 2
%!                       b += 1j * rand(n(j), 3);
%!                   endswitch
%!                   xref = A \ b;
%!                   opt.verbose = 0;
%!                   opt.refine_max_iter = e;
%!                   switch (k)
%!                     case 1
%!                       x = umfpack(A, b, opt);
%!                     otherwise
%!                       x = umfpack(umfpack(A, opt), b);
%!                   endswitch
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(A*x+b));
%!                   assert(max(norm(x - xref, "cols")) <= sqrt(eps) * norm(x));
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   unwind_protect_cleanup
%!     rand("state", state);
%!   end_unwind_protect
%! else
%!   warning("umfpack is not installed");
%! endif
%!error strumpack([]);
%!error strumpack(eye(3));
%!error strumpack(eye(3), zeros(3,1));
%!error strumpack(strumpack(eye(3), struct()));
%!error strumpack(eye(3),zeros(2,1),struct());
%!error strumpack(ones(3,2),zeros(3,1),struct());
