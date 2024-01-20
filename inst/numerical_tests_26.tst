## numerical_tests.tst:26
%!test
%! if (~isempty(which("ndmetis")))
%!   eptr = int32([1, 10, 12]);
%!   eind = int32([2, 30, 1, 40, 9, 7, 55, 80, 77, 13, 5, 100]);
%!   nn = int32(100);
%!   [perm, iperm] = ndmetis(nn, eptr, eind);
%! else
%!   warning("ndmetis was not installed");
%! endif
