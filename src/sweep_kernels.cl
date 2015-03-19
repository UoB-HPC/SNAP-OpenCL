
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Array indexing macros
#define flux_out(a,g,i,j,k,o) flux_out[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)]
#define flux_in(a,g,i,j,k,o) flux_in[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)]
#define source(m,i,j,k,g) source[m+(cmom*i)+(cmom*nx*j)+(cmom*nx*ny*k)+(cmom*nx*ny*nz*g)]
#define flux_i(a,g,j,k) flux_i[a+(nang*g)+(nang*ng*j)+(nang*ng*ny*k)]
#define flux_j(a,g,i,k) flux_j[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*k)]
#define flux_k(a,g,i,j) flux_k[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)]
#define denom(a,g,i,j,k) denom[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)]
#define dd_j(a) dd_j[a]
#define dd_k(a) dd_k[a]
#define mu(a) mu[a]
#define scat_coef(a,m,o) scat_coef[a+(nang*m)+(nang*cmom*o)]
#define time_delta(g) time_delta[g]
#define total_cross_section(g,i,j,k) total_cross_section[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)]
#define scalar(i,j,k,g) scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)]
#define weights(a) weights[a]
#define angular(a,g,i,j,k,o) angular[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)]
#define angular_prev(a,g,i,j,k,o) angular_prev[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)]

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
    __global const double * restrict dd_j,
    __global const double * restrict dd_k,
    __global const double * restrict mu,
    __global const double * restrict scat_coef,
    __global const double * restrict time_delta,
    __global const double * restrict total_cross_section,

    // Angular flux
    __global const double * restrict flux_in,
    __global double * restrict flux_out,

    // Edge fluxes
    __global double * restrict flux_i,
    __global double * restrict flux_j,
    __global double * restrict flux_k,

    // Source
    __global const double * restrict source,
    __global const double * restrict denom
    )
{
    // Get indexes for angle and group
    int a_idx = get_global_id(0) % nang;
    int g_idx = get_global_id(0) / nang;

    if (a_idx >= nang || g_idx >= ng)
        return;

    // Assume transmissive (vacuum boundaries) and that we
    // are sweeping the whole grid so have access to all neighbours
    // This means that we only consider the case for one MPI task
    // at present.

    // Compute angular source
    // Begin with first scattering moment
    double source_term = source(0,i,j,k,g_idx);

    // Add in the anisotropic scattering source moments
    for (unsigned int l = 1; l < cmom; l++)
    {
        source_term += scat_coef(a_idx,l,oct) * source(l,i,j,k,g_idx);
    }

    double psi = source_term + flux_i(a_idx,g_idx,j,k)*mu(a_idx)*dd_i + flux_j(a_idx,g_idx,i,k)*dd_j(a_idx) + flux_k(a_idx,g_idx,i,j)*dd_k(a_idx);

    // Add contribution from last timestep flux if time-dependant
    if (time_delta(g_idx) != 0.0)
    {
        psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k,oct);
    }

    psi *= denom(a_idx,g_idx,i,j,k);

    // Compute upwind fluxes
    double tmp_flux_i = 2.0*psi - flux_i(a_idx,g_idx,j,k);
    double tmp_flux_j = 2.0*psi - flux_j(a_idx,g_idx,i,k);
    double tmp_flux_k = 2.0*psi - flux_k(a_idx,g_idx,i,j);

    // Time differencing on final flux value
    if (time_delta(g_idx) != 0.0)
    {
        psi = 2.0 * psi - flux_in(a_idx,g_idx,i,j,k,oct);
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
        // TODO
        // This while loop causes the the kernel NOT to vectorize in a 1d kernel case for the intel opencl sdk
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
            psi = flux_i(a_idx,g_idx,j,k)*mu(a_idx)*dd_i*(1.0+zeros[0]) + flux_j(a_idx,g_idx,j,k)*dd_j(a_idx)*(1.0+zeros[1]) + flux_k(a_idx,g_idx,i,j)*dd_k(a_idx)*(1.0+zeros[2]);
            if (time_delta(g_idx) != 0.0)
            {
                psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k,oct) * (1.0+zeros[3]);
            }
            psi = 0.5*psi + source_term;
            double recalc_denom = total_cross_section(g_idx,i,j,k);
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
            tmp_flux_i = 2.0 * psi - flux_i(a_idx,g_idx,j,k);
            tmp_flux_j = 2.0 * psi - flux_j(a_idx,g_idx,i,k);
            tmp_flux_k = 2.0 * psi - flux_k(a_idx,g_idx,i,j);
            if (time_delta(g_idx) != 0.0)
            {
                psi = 2.0*psi - flux_in(a_idx,g_idx,i,j,k,oct);
            }
        }
        // Fix up loop is done, just need to set the final values
        tmp_flux_i = tmp_flux_i * zeros[0];
        tmp_flux_j = tmp_flux_j * zeros[1];
        tmp_flux_k = tmp_flux_k * zeros[2];
        psi = psi * zeros[3];
    }

    // Write values to global memory
    flux_i(a_idx,g_idx,j,k) = tmp_flux_i;
    flux_j(a_idx,g_idx,i,k) = tmp_flux_j;
    flux_k(a_idx,g_idx,i,j) = tmp_flux_k;
    flux_out(a_idx,g_idx,i,j,k,oct) = psi;
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
                    tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,o) + angular_prev(a,g,i,j,k,o)));
                }
                else
                {
                    tot_g += weights(a) * angular(a,g,i,j,k,o);
                }
            }
        }
        scalar(i,j,k,g) = tot_g;
    }
}

