## numerical_tests.tst:02
%!test
%! if (~isempty(which("pastix")))
%!   tol = eps^0.35;
%!   rand("seed", 0);
%!   for N=[2, 10, 100]
%!     for i=1:10
%!       for j=1:2
%!         for k=1:2
%!           A = rand(N, N);
%!           b = rand(N, 10);
%!           switch (k)
%!             case 2
%!               A += 1j * rand(N, N);
%!               b += 1j * rand(N, 10);
%!           endswitch
%!           opts.verbose = PASTIX_API_VERBOSE_NOT;
%!           opts.factorization = PASTIX_API_FACT_LU;
%!           opts.matrix_type = PASTIX_API_SYM_NO;
%!           opts.refine_max_iter = int32(10);
%!           opts.check_solution = true;
%!           switch j
%!             case 1
%!               x = pastix(A, b, opts);
%!             case 2
%!               x = pastix(pastix(A, opts), b);
%!           endswitch
%!           f = max(norm(A * x - b, "cols") ./ norm(A * x + b, "cols"));
%!           assert(f <= tol);
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif
