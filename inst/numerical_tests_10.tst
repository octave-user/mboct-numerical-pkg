## numerical_tests.tst:10
%!test
%! if (~isempty(which("mumps")))
%!   rand("seed", 0);
%!   n = [2,4,8,16,32,64,128];
%!   for compl=[false, true]
%!     for e=[0,100]
%!       for u=1:2
%!         for m=[MUMPS_MAT_GEN, MUMPS_MAT_DEF, MUMPS_MAT_SYM]
%!           for i=1:10
%!             for j=1:numel(n)
%!               for k=1:2
%!                 for l=1:2
%!                   for s=1:2
%!                     switch (l)
%!                       case 1
%!                         A = rand(n(j),n(j));
%!                       otherwise
%!                         A = sprand(n(j), n(j), 0.1) + diag(rand(n(j),1));
%!                     endswitch
%!                     if (compl)
%!                       A *= (rand() + 1j * rand());
%!                     endif
%!                     Af = A;
%!                     b = rand(n(j), 3);
%!                     switch (s)
%!                       case {1,2}
%!                         switch (s)
%!                           case 2
%!                             A += A.';
%!                           case 1
%!                             A *= A.';
%!                             A += eye(size(A));
%!                         endswitch
%!                         Af = A;
%!                         if (s == 2 && m > 0)
%!                           [r, c, d] = find(A);
%!                           if (u == 1)
%!                             idx = find(r >= c);
%!                           else
%!                             idx = find(r <= c);
%!                           endif
%!                           r = r(idx);
%!                           c = c(idx);
%!                           d = d(idx);
%!                           Af = sparse(r, c, d, n(j), n(j));
%!                           if (l == 1)
%!                             Af = full(Af);
%!                           endif
%!                         endif
%!                     endswitch
%!                     xref = A \ b;
%!                     opt.verbose = MUMPS_VER_WARN;
%!                     opt.refine_max_iter = e;
%!                     opt.matrix_type = m;
%!                     switch (k)
%!                       case 1
%!                         x = mumps(Af, b, opt);
%!                       otherwise
%!                         x = mumps(mumps(Af, opt), b);
%!                     endswitch
%!                     assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(b));
%!                     assert(max(norm(A * x - b, "cols")) <= sqrt(eps) * norm(A*x+b));
%!                     assert(max(norm(x - xref, "cols")) <= sqrt(eps) * norm(x));
%!                   endfor
%!                 endfor
%!               endfor
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! else
%!   warning("mumps is not installed");
%! endif
