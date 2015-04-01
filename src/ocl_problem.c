#include "ocl_problem.h"

void set_ocl_problem_(
    int *nx_, int *ny_, int *nz_,
    int *ng_, int *nang_, int *noct_, int *cmom_,
    int *ichunk_,
    double *dt_,
    int *timesteps_, int *outers_, int *inners_)
{
    // Save problem size information to globals
    nx = *nx_;
    ny = *ny_;
    nz = *nz_;
    ng = *ng_;
    nang = *nang_;
    noct = *noct_;
    cmom = *cmom_;
    ichunk = *ichunk_;
    dt = *dt_;
    timesteps = *timesteps_;
    outers = *outers_;
    inners = *inners_;
}
