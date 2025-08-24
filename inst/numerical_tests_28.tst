%!test
%! try
%! state = rand("state");
%! unwind_protect
%!  rand("seed", 0);
%!  N = 1000;
%!  a = rand(3, N);
%!  b = rand(3, N);
%!  c = cross(a, b);
%!  c1 = c2 = zeros(3, N);
%!  for i=1:N
%!    c1(:, i) = skew(a(:, i)) * b(:, i);
%!    c2(:, i) = -skew(b(:, i)) * a(:, i);
%!  endfor
%!  assert(c1, c);
%!  assert(c2, c);
%! unwind_protect_cleanup
%!  rand("state", state);
%! end_unwind_protect
%! catch
%!   gtest_error = lasterror();
%!   gtest_fail(gtest_error, evalin("caller", "__file"));
%!   rethrow(gtest_error);
%! end_try_catch
