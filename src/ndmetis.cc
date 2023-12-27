// Copyright (C) 2018(-2023) Reinhard <octave-user@a1.net>

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; If not, see <http://www.gnu.org/licenses/>.

#include "config.h"

#include <octave/oct.h>
#include <metis.h>

// PKG_ADD: autoload ("ndmetis", "__mboct_numerical__.oct");
// PKG_DEL: autoload ("ndmetis", "__mboct_numerical__.oct", "remove");

DEFUN_DLD (ndmetis, args, nargout,
           "-*- texinfo -*-\n"
           "@deftypefn {Function File} [@var{perm}, @var{iperm}] = ndmetis(@var{nn}, @var{eptr}, @var{eind})\n\n"
           "Compute a permutation vector for fill-in reduction of a Finite Element mesh using METIS.\n\n"
           "@var{nn} @dots{} The number of nodes\n\n"
           "@var{eptr} @dots{} Element pointer\n\n"
           "@var{eind} @dots{} Element index\n\n"
           "@end deftypefn\n")
{
    octave_value_list retval;

    if (args.length() != 3 || nargout > 2) {
        print_usage();
        return retval;
    }

    idx_t nn = args(0).int_value();

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif

    intNDArray<idx_t> eptr(args(1).int32_array_value());

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif

    intNDArray<idx_t> eind(args(2).int32_array_value());

#if OCTAVE_MAJOR_VERSION < 6
    if (error_state) {
        return retval;
    }
#endif

    intNDArray<idx_t> perm(dim_vector(nn, 1)), iperm(dim_vector(nn, 1));

    if (eptr.numel() == 0 || eind.numel() == 0) {
        for (octave_idx_type i = 0; i < nn; ++i) {
            perm(i) = i + 1;
            iperm(i) = i + 1;
        }
    } else {
        for (octave_idx_type i = 0; i < eptr.numel(); ++i) {
            if (eptr(i) < 1 || eptr(i) > eind.numel()) {
                 error("invalid element index in eptr(%ld)", static_cast<long>(i));
                return retval;
            }

            if (i > 0 && eptr(i) <= eptr(i - 1)) {
                 error("invalid element index in eptr(%ld)", static_cast<long>(i));
                return retval;
            }
        }

        for (octave_idx_type i = 0; i < eind.numel(); ++i) {
            if (eind(i) < 1 || eind(i) > nn) {
                 error("invalid node index in eind(%ld)", static_cast<long>(i));
                return retval;
            }
        }

        idx_t ne = eptr.numel() - 1;
        idx_t* xadj = nullptr;
        idx_t* adjncy = nullptr;
        idx_t options[METIS_NOPTIONS];

        METIS_SetDefaultOptions(options);

        options[METIS_OPTION_NUMBERING] = 1;

        int status = METIS_MeshToNodal(&ne,
                                       &nn,
                                       eptr.fortran_vec(),
                                       eind.fortran_vec(),
                                       &options[METIS_OPTION_NUMBERING],
                                       &xadj,
                                       &adjncy);

        if (status != METIS_OK) {
            error("METIS_MeshToNodal failed with status %d", status);
            goto exit_handler;
        }

        status = METIS_NodeND(&nn,
                              xadj,
                              adjncy,
                              nullptr,
                              options,
                              perm.fortran_vec(),
                              iperm.fortran_vec());

        if (status != METIS_OK) {
            error("METIS_NodeND failed with status %d", status);
            goto exit_handler;
        }

    exit_handler:
        METIS_Free(xadj);
        METIS_Free(adjncy);
    }

    retval.append(int32NDArray(perm));
    retval.append(int32NDArray(iperm));

    return retval;
}
