## numerical_tests.tst:09
%!test
%! if (~isempty(which("pastix")))
%!   rand("seed", 0);
%!   for mt=[PASTIX_API_SYM_YES]
%!     for s=0:2
%!       for f=[PASTIX_API_FACT_LLT, PASTIX_API_FACT_LDLT]
%!         for N=[2, 10, 100]
%!           for i=1:10
%!             for k=1:2
%!               A = rand(N, N)+2 * diag(rand(N,1));
%!               switch (k)
%!                 case 2
%!                   A += 1j * (rand(N, N)+2 * diag(rand(N,1)));
%!               endswitch
%!               switch mt
%!                 case PASTIX_API_SYM_NO
%!                 case PASTIX_API_SYM_YES
%!                   switch f
%!                     case PASTIX_API_FACT_LLT
%!                       A *= A.';
%!                     case PASTIX_API_FACT_LDLT
%!                       A *= A.';
%!                   endswitch
%!               endswitch
%!               [r, c, d] = find(A);
%!               opts.factorization = f;
%!               switch s
%!                 case 0
%!                   idx = find(r >= c);
%!                 case 1
%!                   idx = find(r <= c);
%!                 otherwise
%!                   idx = 1:numel(r);
%!               endswitch
%!               b = rand(N, 10);
%!               opts.verbose = PASTIX_API_VERBOSE_NOT;
%!               opts.refine_max_iter = int32(10);
%!               opts.matrix_type = mt;
%!               opts.number_of_threads = 1;
%!               opts.check_solution = false;
%!               opts.epsilon_refinement = eps^0.9;
%!               Asym = sparse(r(idx), c(idx), d(idx));
%!               assert(nnz(Asym) > 0);
%!               x = pastix(Asym, b, opts);
%!               xref = A \ b;
%!               tol = eps^0.3;
%!               fpas = norm(A * x - b, "cols") ./ norm(A * x + b, "cols");
%!               fref = norm(A * xref - b, "cols") ./ norm(A * xref + b, "cols");
%!               assert(max(fpas) <= tol);
%!               assert(max(fref) <= tol);
%!             endfor
%!           endfor
%!         endfor
%!       endfor
%!     endfor
%!   endfor
%! endif
%!error mumps([]);
%!error mumps(eye(3));
%!error mumps(eye(3), zeros(3,1));
%!error mumps(mumps(eye(3), struct()));
%!error mumps(eye(3),zeros(2,1),struct());
%!error mumps(ones(3,2),zeros(3,1),struct());
