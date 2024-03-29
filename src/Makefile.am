##  Copyright (C) 2018(-2021) Reinhard <octave-user@a1.net>

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; If not, see <http://www.gnu.org/licenses/>.

oct_OBJ:=sp_sym_mtimes.@OBJEXT@

if USE_PASTIX
  CPPFLAGS+=$(PASTIX_INC)
  LIBS+=$(PASTIX_LIBS)
  oct_OBJ+=pastix.@OBJEXT@
endif

if USE_MUMPS
  CPPFLAGS+=$(MUMPS_INC)
  LIBS+=$(MUMPS_LIBS)
  oct_OBJ+=mumps.@OBJEXT@
endif

if USE_ARPACK
  LIBS+=$(ARPACK_LIBS)
  oct_OBJ+=eig_sym.@OBJEXT@
endif

if USE_UMFPACK
  CPPFLAGS+=$(UMFPACK_INC)
  LIBS+=$(UMFPACK_LIBS)
  oct_OBJ+=umfpack.@OBJEXT@
endif

if USE_METIS
  CPPFLAGS+=$(METIS_INC)
  LIBS+=$(METIS_LIBS)
  oct_OBJ+=ndmetis.@OBJEXT@
endif

if USE_DSPEV
  LIBS+=$(LAPACK_LIBS)
  oct_OBJ+=dspev.@OBJEXT@
endif

if USE_DSBGVX
  LIBS+=$(LAPACK_LIBS)
  oct_OBJ+=dsbgvx.@OBJEXT@
endif

if USE_STRUMPACK
  CPPFLAGS+=$(STRUMPACK_INC)
  LIBS+=$(STRUMPACK_LIBS)
  oct_OBJ+=strumpack.@OBJEXT@
endif

if USE_PARDISO
  CPPFLAGS+=$(PARDISO_INC)
  LIBS+=$(PARDISO_LIBS)
  oct_OBJ+=pardiso.@OBJEXT@
endif

oct_FILE:=__mboct_numerical__.oct

bin_FILES:=$(oct_FILE)

all-local: $(bin_FILES)

clean-local:
	rm -f $(bin_FILES) $(oct_OBJ)

distclean-local:
	rm -f oct-alt-includes.h

%.@OBJEXT@: %.cc
	$(MKOCTFILE) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(oct_FILE): $(oct_OBJ)
	$(MKOCTFILE) $^ $(LDFLAGS) $(LIBS) -o $@

.PHONY: clean-local distclean-local
