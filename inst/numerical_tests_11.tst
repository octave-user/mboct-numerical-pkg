## numerical_tests.tst:11
%!test
%! try
%! if (~isempty(which("ndmetis")))
%!   eptr = int32([1, 10, 12]);
%!   eind = int32([2, 30, 1, 40, 9, 7, 55, 80, 77, 13, 5, 100]);
%!   nn = int32(100);
%!   [perm, iperm] = ndmetis(nn, eptr, eind);
%! else
%!   warning("ndmetis is not installed");
%! endif
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
%!error umfpack([]);
%!error umfpack(eye(3));
%!error umfpack(eye(3), zeros(3,1));
%!error umfpack(umfpack(eye(3), struct()));
%!error umfpack(eye(3),zeros(2,1),struct());
%!error umfpack(ones(3,2),zeros(3,1),struct());
