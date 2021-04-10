## Copyright (C) 2011(-2021) Reinhard <octave-user@a1.net>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} @var{A} = skew(@var{v})
## Return a skew symmetric cross product matrix @var{A} such that skew(a) * b == cross(a, b)
## @end deftypefn

function A = skew(a)
  if (~(nargin == 1 && isvector(a) && numel(a) == 3))
    print_usage();
  endif
    
  A = [  0,     -a(3),   a(2);
         a(3),   0,     -a(1);
        -a(2),   a(1),   0     ];
endfunction

%!test
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
