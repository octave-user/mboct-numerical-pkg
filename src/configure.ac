dnl  Copyright (C) 2018(-2020) Reinhard <octave-user@a1.net>

dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3 of the License, or
dnl (at your option) any later version.

dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.

dnl You should have received a copy of the GNU General Public License
dnl along with this program; If not, see <http://www.gnu.org/licenses/>.

AC_PREREQ([2.69])
AC_INIT([mboct-numerical-pkg], [0.1.0])
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_MACRO_DIRS([m4])

AM_INIT_AUTOMAKE

AC_PROG_CXX

AC_ARG_VAR([PKG_CONFIG],[Program to return meta information about installed libraries])
AC_ARG_VAR([PASTIX_INC],[PaStiX include flags])
AC_ARG_VAR([PASTIX_LIBS],[PaStiX libraries])
AC_ARG_VAR([MUMPS_INC],[MUMPS include flags])
AC_ARG_VAR([MUMPS_LIBS],[MUMPS libraries])
AC_ARG_VAR([ARPACK_LIBS],[ARPACK libraries])
AC_ARG_VAR([UMFPACK_INC],[Umfpack include flags])
AC_ARG_VAR([UMFPACK_LIBS],[Umfpack libraries])
AC_ARG_VAR([METIS_INC],[Metis include flags])
AC_ARG_VAR([METIS_LIBS],[Metis libraries])
AC_ARG_VAR([LAPACK_LIBS],[Lapack libraries])

AC_CHECK_PROG([PKG_CONFIG], [pkg-config], [pkg-config])
AC_CHECK_PROG([MKOCTFILE], [mkoctfile], [mkoctfile])
AC_CHECK_PROG([OCTAVE_CONFIG], [octave-config], [octave-config])

if test -z "$MKOCTFILE"; then
  AC_MSG_ERROR([mkoctfile not found], 1);
fi

if test -z "$OCTAVE_CONFIG"; then
  AC_MSG_ERROR([octave-config not found], 1);
fi

if test -z "${PASTIX_INC}"; then
  if ! test -z "${PKG_CONFIG}"; then
    PASTIX_INC="`${PKG_CONFIG} --cflags pastix`"
  fi
fi

if test -z "${PASTIX_LIBS}"; then
  if ! test -z "${PKG_CONFIG}"; then
    PASTIX_LIBS="`${PKG_CONFIG} --libs pastix`"
  else
    PASTIX_LIBS="-lpastix -lpastix_kernels -lspm -lhwloc"
  fi
fi

MUMPS_LIBS="${MUMPS_LIBS:--ldmumps_seq -lmpiseq_seq-4.10.0 -lgfortran}"
ARPACK_LIBS="${ARPACK_LIBS:--larpack}"
UMFPACK_LIBS="${UMFPACK_LIBS:--lumfpack}"
METIS_LIBS="${METIS_LIBS:--lmetis}"
LAPACK_LIBS="${LAPACK_LIBS:-`${MKOCTFILE} -p LAPACK_LIBS`}"
LAPACK_LIBS="${LAPACK_LIBS:--llapack}"

AC_LANG(C++)

AC_CHECK_HEADERS([cmath cstdint cstring stdexcept algorithm array iostream], [], AC_MSG_ERROR([standard C++ headers not found], 1))

TCPPFLAGS="${CPPFLAGS}"
TLIBS="${LIBS}"

CPPFLAGS="${PASTIX_INC} ${CPPFLAGS}"
LIBS="${PASTIX_LIBS} ${LIBS}"

AC_LANG(C)
AC_CHECK_HEADER([pastix.h])

AC_CHECK_FUNCS([pastixInit \
                pastixInitParam \
                pastix_task_analyze \
                pastix_task_numfact \
                pastix_task_solve \
                pastix_task_refine \
                pastixFinalize \
                spmNorm \
                spmScalMatrix \
                spmScalVector \
                spmExit \
                spmCheckAxb],[have_pastix_6=yes],[have_pastix_6=no])

AC_CHECK_FUNCS([pastix \
                pastix_checkMatrix],[have_pastix_5=yes],[have_pastix_5=no])

AM_CONDITIONAL([USE_PASTIX],[test "${ac_cv_header_pastix_h}" = yes && ( test "${have_pastix_6}" = "yes" || test "${have_pastix_5}" = "yes" )])

if test "${have_pastix_6}" = "yes"; then
   AC_DEFINE([USE_PASTIX_6],1,[Define if the PaStiX interface from version 6.0 or later can be used])
else
   if test "${have_pastix_5}" = "yes"; then
      AC_DEFINE([USE_PASTIX_5],1,[Define if the PaStiX interface up to version 5.3 can be used])
   fi
fi

CPPFLAGS="${TCPPFLAGS}"
LIBS="${TLIBS}"

TCPPFLAGS="${CPPFLAGS}"
TLIBS="${LIBS}"

CPPFLAGS="${MUMPS_INC} ${CPPFLAGS}"
LIBS="${MUMPS_LIBS} ${LIBS}"

AC_LANG(C)

AC_CHECK_HEADERS([dmumps_c.h mumps_seq/mpi.h],[have_dmumps_c_h=yes],[have_dmumps_c_h=no])

AC_CHECK_FUNCS([dmumps_c \
                MPI_Init \
                MPI_Comm_rank \
                MPI_Finalize],[have_dmumps_c_f=yes],[have_dmumps_c_f=no])

AM_CONDITIONAL([USE_MUMPS],[test "${have_dmumps_c_h}" = yes && test "${have_dmumps_c_f}" == "yes"])

CPPFLAGS="${TCPPFLAGS}"
LIBS="${TLIBS}"

TCPPFLAGS="${CPPFLAGS}"
TLLIBS="${LIBS}"

CPPFLAGS="${UMFPACK_INC} ${CPPFLAGS}"
LIBS="${UMFPACK_LIBS} ${LIBS}"

AC_CHECK_HEADERS([suitesparse/umfpack.h])

AC_CHECK_FUNCS([umfpack_dl_defaults \
                umfpack_dl_symbolic \
                umfpack_dl_report_info \
                umfpack_dl_report_status \
                umfpack_dl_numeric \
                umfpack_dl_solve \
                umfpack_dl_free_symbolic \
                umfpack_dl_free_numeric],[have_umfpack=yes],[have_umfpack=no])

AM_CONDITIONAL([USE_UMFPACK],[test "${ac_cv_header_suitesparse_umfpack_h}" = "yes" && test "${have_umfpack}" = "yes"])

CPPFLAGS="${TCPPFLAGS}"
LIBS="${TLIBS}"

CPPFLAGS="${METIS_INC} ${CPPFLAGS}"
LIBS="${METIS_LIBS} ${LIBS}"

AC_CHECK_HEADERS([metis.h])

AC_CHECK_FUNCS([METIS_SetDefaultOptions \
                METIS_MeshToNodal \
                METIS_NodeND \
                METIS_Free],[have_metis=yes],[have_metis=no])

AM_CONDITIONAL([USE_METIS],[test "${ac_cv_header_metis_h}" = "yes" && test "${have_metis}" = "yes"])

CPPFLAGS="${TCPPFLAGS}"
LIBS="${TLIBS}"

TLIBS="${LIBS}"

LIBS="${LAPACK_LIBS} ${LIBS}"

AC_CHECK_FUNCS([dspev dspev_ DSPEV DSPEV_])

AM_CONDITIONAL([USE_DSPEV], [test "${ac_cv_func_dspev}" = "yes" || \
                             test "${ac_cv_func_dspev_}" = "yes" || \
                             test "${ac_cv_func_DSPEV}" = "yes" || \
                             test "${ac_cv_func_DSPEV_}" = "yes"])
               
AC_CHECK_FUNCS([dsbgvx dsbgvx_ DSBGVX DSBGVX_ dlamch dlamch_ DLAMCH DLAMCH_])

AM_CONDITIONAL([USE_DSBGVX], [(test "${ac_cv_func_dsbgvx}" = "yes" || \
                               test "${ac_cv_func_dsbgvx_}" = "yes" || \
                               test "${ac_cv_func_DSBGVX}" = "yes" || \
                               test "${ac_cv_func_DSBGVX_}" = "yes") &&
                               test "${ac_cv_func_dlamch}" = "yes" ||
                               test "${ac_cv_func_dlamch_}" = "yes" ||
                               test "${ac_cv_func_DLAMCH}" = "yes" ||
                               test "${ac_cv_func_DLAMCH_}" = "yes"])

LIBS="${TLIBS}"

OCTINCLUDEDIR="${OCTINCLUDEDIR:-`${MKOCTFILE} -p INCFLAGS`}"
OCTLIBDIR="${OCTLIBDIR:-`${MKOCTFILE} -p OCTLIBDIR`}"
OCTLIBS="${OCTLIBS:-`${MKOCTFILE} -p LIBOCTAVE` `${MKOCTFILE} -p LIBOCTINTERP`}"
CXX="${CXX:-`${MKOCTFILE} -p CXX`}"
CXXFLAGS="${CXXFLAGS} `${MKOCTFILE} -p CXXFLAGS`"

dnl Force mkoctfile to link to Octave's libraries in order to avoid unresolved symbols at runtime
OCTLDFLAGS="${OCTLDFLAGS:--L${OCTLIBDIR} -Wl,-rpath=${OCTLIBDIR} -Wl,--no-undefined}"

TCPPFLAGS="${CPPFLAGS}"
TLDFLAGS="${LDFLAGS}"
TLIBS="${LIBS}"

CPPFLAGS="${OCTINCLUDEDIR} ${CPPFLAGS}"
LDFLAGS="${OCTLDFLAGS} ${LDFLAGS}"
LIBS="${OCTLIBS} ${LIBS}"

AC_LANG(C++)
AC_CHECK_HEADERS(octave/oct.h octave/interpreter.h octave/parse.h, [], AC_MSG_ERROR([octave header files not found], 1))

AC_MSG_CHECKING([for octave libraries])
AC_TRY_LINK([
  #include <octave/oct.h>
],[
  octave_value_list ovl;
  ovl.append(octave_value(1.5));
],[use_octlibs=yes],[use_octlibs=no])

AC_MSG_RESULT([${use_octlibs}])

if ! test "${use_octlibs}" = "yes"; then
  LDFLAGS="${TLDFLAGS}"
  LIBS="${TLIBS}"
fi

TLIBS="${LIBS}"
LIBS="${ARPACK_LIBS} ${LIBS}"

AC_CHECK_HEADERS(octave/lo-arpack-proto.h)

AC_MSG_CHECKING([for dsaupd])
AC_TRY_LINK([
#define HAVE_ARPACK 1
#include <octave/lo-arpack-proto.h>
],[
        F77_INT ido, n, nev, ncv, *iparam, *ipntr, lwork, info;
        F77_DBLE tol, *presid, *v, *workdp, *worklp;
        const char bmat[] = "G", type[] = "LM";

        F77_FUNC (dsaupd, DSAUPD)
            (ido,
             F77_CONST_CHAR_ARG2 (bmat, 1),
             n,
             F77_CONST_CHAR_ARG2 (type, 2),
             nev,
             tol,
             presid,
             ncv,
             v,
             n,
             iparam,
             ipntr,
             workdp,
             worklp,
             lwork,
             info
             F77_CHAR_ARG_LEN(1)
             F77_CHAR_ARG_LEN(2));
             ],[have_dsaupd=yes],[have_dsaupd=no])

AC_MSG_RESULT([$have_dsaupd])

AC_MSG_CHECKING([for dseupd])

AC_TRY_LINK([
#define HAVE_ARPACK 1
#include <octave/lo-arpack-proto.h>
],[
    bool rvec;
    F77_DBLE tol, *presid, *d, *v, *workdp, *worklp, sigma;
    F77_INT n, nev, ncv, *sel, *iparam, *ipntr, lwork, info;
    const char bmat[] = "G", type[] = "SM";
    F77_FUNC (dseupd, DSEUPD)
        (rvec,
         F77_CONST_CHAR_ARG2 ("A", 1),
         sel,
         d,
         v,
         n,
         sigma,
         F77_CONST_CHAR_ARG2 (bmat, 1),
         n,
         F77_CONST_CHAR_ARG2 (type, 2),
         nev,
         tol,
         presid,
         ncv,
         v,
         n,
         iparam,
         ipntr,
         workdp,
         worklp,
         lwork,
         info
         F77_CHAR_ARG_LEN(1)
         F77_CHAR_ARG_LEN(1)
         F77_CHAR_ARG_LEN(2));
],[have_dseupd=yes],[have_dseupd=no])

AC_MSG_RESULT([$have_dseupd])

AM_CONDITIONAL([USE_ARPACK],[test "${have_dsaupd}" = "yes" && test "${have_dseupd}" = "yes" && test "${ac_cv_header_octave_lo_arpack_proto_h}" = "yes"])

LIBS="${TLIBS}"

## Simple symbol alternatives of different Octave versions.
OF_OCTAVE_LIST_ALT_SYMS([
[dnl
  [feval],
  [octave::feval],
  [[octave::feval ("date");]],
  [OCTAVE__FEVAL],
  [[#include <octave/parse.h>]],
  [[#include <octave/parse.h>]]
],

[dnl
  [is_symmetric],
  [issymmetric],
  [[SparseMatrix ().issymmetric ();]],
  [OV_ISSYMMETRIC],
  [],
  []
]

],
   [oct-alt-includes.h])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
