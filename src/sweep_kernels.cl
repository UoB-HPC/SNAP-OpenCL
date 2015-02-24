
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Array indexing macros
#define flux_out(a,i,j,k,o,g) flux_out[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)]
#define flux_in(a,i,j,k,o,g) flux_in[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)]
#define source(m,i,j,k,g) source[m+(cmom*i)+(cmom*nx*j)+(cmom*nx*ny*k)+(cmom*nx*ny*nz*g)]
#define flux_i(a,j,k,g) flux_i[a+(nang*j)+(nang*ny*k)+(nang*ny*nz*g)]
#define flux_j(a,i,k,g) flux_j[a+(nang*i)+(nang*nx*k)+(nang*nx*nz*g)]
#define flux_k(a,i,j,g) flux_k[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*g)]
#define denom(a,i,j,k,g) denom[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*g)]
#define dd_j(a) dd_j[a]
#define dd_k(a) dd_k[a]
#define mu(a) mu[a]
#define scat_coef(a,m,o) scat_coef[a+(nang*m)+(nang*cmom*o)]
#define time_delta(g) time_delta[g]
#define total_cross_section(i,j,k,g) total_cross_section[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)]
#define scalar(i,j,k,g) scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)]
#define weights(a) weights[a]
#define angular(a,i,j,k,o,g) angular[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)]
#define angular_prev(a,i,j,k,o,g) angular_prev[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)]

// Solve the transport equations for a single angle in a single cell for a single group
__kernel void sweep_cell(
    // Current cell index
    const unsigned int i,
    const unsigned int j,
    const unsigned int k,
    const unsigned int oct,

    // Problem sizes
    const unsigned int ichunk,
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int nang,
    const unsigned int noct,
    const unsigned int cmom,

    // Coefficients
    const double dd_i,
    __global double *dd_j,
    __global double *dd_k,
    __global double *mu,
    __global double *scat_coef,
    __global double *time_delta,
    __global double *total_cross_section,

    // Angular flux
    __global double *flux_in,
    __global double *flux_out,

    // Edge fluxes
    __global double *flux_i,
    __global double *flux_j,
    __global double *flux_k,

    // Source
    __global double *source,
    __global double *denom
    )
{
    // Get indexes for angle and group
    int a_idx = get_global_id(0);
    int g_idx = get_global_id(1);

    // Assume transmissive (vacuum boundaries) and that we
    // are sweeping the whole grid so have access to all neighbours
    // This means that we only consider the case for one MPI task
    // at present.

    // Compute angular source
    // Begin with first scattering moment)
    double source_term = source(0,i,j,k,g_idx);

    // Add in the anisotropic scattering source moments
    for (unsigned int l = 1; l < cmom; l++)
    {
        source_term += scat_coef(a_idx,l,oct) * source(l,i,j,k,g_idx);
    }

    double psi = source_term + flux_i(a_idx,j,k,g_idx)*mu(a_idx)*dd_i + flux_j(a_idx,i,k,g_idx)*dd_j(a_idx) + flux_k(a_idx,i,j,g_idx)*dd_k(a_idx);

    // Add contribution from last timestep flux if time-dependant
    if (time_delta(g_idx) != 0.0)
    {
        psi += time_delta(g_idx) * flux_in(a_idx,i,j,k,oct,g_idx);
    }

    psi *= denom(a_idx,i,j,k,g_idx);

    // Compute upwind fluxes
    double tmp_flux_i = 2.0*psi - flux_i(a_idx,j,k,g_idx);
    double tmp_flux_j = 2.0*psi - flux_j(a_idx,i,k,g_idx);
    double tmp_flux_k = 2.0*psi - flux_k(a_idx,i,j,g_idx);

    // Time differencing on final flux value
    if (time_delta(g_idx) != 0.0)
    {
        psi = 2.0 * psi - flux_in(a_idx,i,j,k,oct,g_idx);
    }

    // Perform the fixup loop
    if (
            tmp_flux_i < 0.0 ||
            tmp_flux_j < 0.0 ||
            tmp_flux_k < 0.0 ||
            psi < 0.0)
    {
        double zeros[4] = {1.0, 1.0, 1.0, 1.0};
        int num_to_fix = 4;
        while (
            tmp_flux_i < 0.0 ||
            tmp_flux_j < 0.0 ||
            tmp_flux_k < 0.0 ||
            psi < 0.0)
        {


            // Record which ones are zero
            if (tmp_flux_i < 0.0) zeros[0] = 0.0;
            if (tmp_flux_j < 0.0) zeros[1] = 0.0;
            if (tmp_flux_k < 0.0) zeros[2] = 0.0;
            if (psi < 0.0) zeros[3] = 0.0;

            if (num_to_fix == zeros[0] + zeros[1] + zeros[2] + zeros[3])
            {
                // We have fixed up enough
                break;
            }
            num_to_fix = zeros[0] + zeros[1] + zeros[2] + zeros[3];

            // Recompute cell centre value
            psi = flux_i(a_idx,j,k,g_idx)*mu(a_idx)*dd_i*(1.0+zeros[0]) + flux_j(a_idx,j,k,g_idx)*dd_j(a_idx)*(1.0+zeros[1]) + flux_k(a_idx,i,j,g_idx)*dd_k(a_idx)*(1.0+zeros[2]);
            if (time_delta(g_idx) != 0.0)
            {
                psi += time_delta(g_idx) * flux_in(a_idx,i,j,k,oct,g_idx) * (1.0+zeros[3]);
            }
            psi = 0.5*psi + source_term;
            double recalc_denom = total_cross_section(i,j,k,g_idx);
            recalc_denom += mu(a_idx) * dd_i * zeros[0];
            recalc_denom += dd_j(a_idx) * zeros[1];
            recalc_denom += dd_k(a_idx) * zeros[2];
            recalc_denom += time_delta(g_idx) * zeros[3];

            if (recalc_denom > 1.0E-12)
            {
                psi /= recalc_denom;
            }
            else
            {
                psi = 0.0;
            }

            // Recompute the edge fluxes with the new centre value
            tmp_flux_i = 2.0 * psi - flux_i(a_idx,j,k,g_idx);
            tmp_flux_j = 2.0 * psi - flux_j(a_idx,i,k,g_idx);
            tmp_flux_k = 2.0 * psi - flux_k(a_idx,i,j,g_idx);
            if (time_delta(g_idx) != 0.0)
            {
                psi = 2.0*psi - flux_in(a_idx,i,j,k,oct,g_idx);
            }
        }
        // Fix up loop is done, just need to set the final values
        tmp_flux_i = tmp_flux_i * zeros[0];
        tmp_flux_j = tmp_flux_j * zeros[1];
        tmp_flux_k = tmp_flux_k * zeros[2];
        psi = psi * zeros[3];
    }

    // Write values to global memory
    flux_i(a_idx,j,k,g_idx) = tmp_flux_i;
    flux_j(a_idx,i,k,g_idx) = tmp_flux_j;
    flux_k(a_idx,i,j,g_idx) = tmp_flux_k;
    flux_out(a_idx,i,j,k,oct,g_idx) = psi;
    return;
}


// Compute the scalar flux from the angular flux
// Each work item is assigned a single cell
// The work item loops over the energy groups
// and the angles to create a single value
// for each energy group per cell
__kernel void reduce_angular(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int noct,
    __global double *weights,
    __global double *angular,
    __global double *angular_prev,
    __global double *time_delta,
    __global double *scalar)
{
    // Cell index
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // For groups
    for (unsigned int g = 0; g < ng; g++)
    {
        double tot_g = 0.0;
        // For octants
        for (unsigned int o = 0; o < noct; o++)
        {
            // For angles
            for (unsigned int a = 0; a < nang; a++)
            {
                // NOTICE: we do the reduction with psi, not ptr_out.
                // This means that (line 307) the time dependant
                // case isnt the value that is summed, but rather the
                // flux in the cell
                // Note all work items will all take the same branch
                if (time_delta(g) != 0.0)
                {
                    tot_g += weights(a) * (0.5 * (angular(a,i,j,k,o,g) + angular_prev(a,i,j,k,o,g)));
                }
                else
                {
                    tot_g += weights(a) * angular(a,i,j,k,o,g);
                }
            }
        }
        scalar(i,j,k,g) = tot_g;
    }
}

