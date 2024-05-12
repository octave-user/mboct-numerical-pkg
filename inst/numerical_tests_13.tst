## numerical_tests.tst:13
%!test
%! try
%! if (~isempty(which("strumpack")))
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
%!                       x = strumpack(A, b, opt);
%!                     otherwise
%!                       x = strumpack(strumpack(A, opt), b);
%!                   endswitch
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                   assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * max(norm(A*x+b, "cols")));
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
%!   warning("strumpack is not installed");
%! endif
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
