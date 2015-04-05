
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Array indexing macros
#define flux_out(a,g,i,j,k) flux_out[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define flux_in(a,g,i,j,k) flux_in[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define source(m,i,j,k,g) source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define flux_i(a,g,j,k) flux_i[(a)+(nang*(g))+(nang*ng*(j))+(nang*ng*ny*(k))]
#define flux_j(a,g,i,k) flux_j[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(k))]
#define flux_k(a,g,i,j) flux_k[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))]
#define denom(a,g,i,j,k) denom[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define dd_j(a) dd_j[(a)]
#define dd_k(a) dd_k[(a)]
#define mu(a) mu[(a)]
#define eta(a) eta[(a)]
#define xi(a) xi[(a)]
#define scat_coef(a,m,o) scat_coef[(a)+(nang*(m))+(nang*cmom*(o))]
#define time_delta(g) time_delta[(g)]
#define total_cross_section(g,i,j,k) total_cross_section[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define scalar(i,j,k,g) scalar[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define weights(a) weights[(a)]

#define angular(a,g,i,j,k,o) angular##o[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev(a,g,i,j,k,o) angular_prev##o[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define velocity(g) velocity[(g)]

#define map(i,j,k) map[(i)+(nx*(j))+(nx*ny*(k))]
#define xs(i,g) xs[(i)+(nmat*(g))]

#define g2g_source(m,i,j,k,g) g2g_source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define fixed_source(i,j,k,g) fixed_source[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define gg_cs(m,l,g1,g2) gg_cs[(m)+(nmat*(l))+(nmat*nmom*(g1))+(nmat*nmom*ng*(g2))]
#define lma(m) lma[(m)]
#define scalar_mom(m,i,j,k,g) scalar_mom[(m)+((cmom-1)*(i))+((cmom-1)*nx*(j))+((cmom-1)*nx*ny*(k))+((cmom-1)*nx*ny*nz*(g))]

#define scat_cs(m,i,j,k,g) scat_cs[(m)+(nmom*(i))+(nmom*nx*(j))+(nmom*nx*ny*(k))+(nmom*nx*ny*nz*(g))]


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
        psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k);
    }

    psi *= denom(a_idx,g_idx,i,j,k);

    // Compute upwind fluxes
    double tmp_flux_i = 2.0*psi - flux_i(a_idx,g_idx,j,k);
    double tmp_flux_j = 2.0*psi - flux_j(a_idx,g_idx,i,k);
    double tmp_flux_k = 2.0*psi - flux_k(a_idx,g_idx,i,j);

    // Time differencing on final flux value
    if (time_delta(g_idx) != 0.0)
    {
        psi = 2.0 * psi - flux_in(a_idx,g_idx,i,j,k);
    }

    // Perform the fixup loop
    double zeros[4] = {1.0, 1.0, 1.0, 1.0};
    int num_to_fix = 4;
    // Fixup is a bounded loop as we will worst case fix up each face and centre value one after each other
    for (int fix = 0; fix < 4; fix++)
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
            psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k) * (1.0+zeros[3]);
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
            psi = 2.0*psi - flux_in(a_idx,g_idx,i,j,k);
        }
    }
    // Fix up loop is done, just need to set the final values
    tmp_flux_i = tmp_flux_i * zeros[0];
    tmp_flux_j = tmp_flux_j * zeros[1];
    tmp_flux_k = tmp_flux_k * zeros[2];
    psi = psi * zeros[3];

    // Write values to global memory
    flux_i(a_idx,g_idx,j,k) = tmp_flux_i;
    flux_j(a_idx,g_idx,i,k) = tmp_flux_j;
    flux_k(a_idx,g_idx,i,j) = tmp_flux_k;
    flux_out(a_idx,g_idx,i,j,k) = psi;
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
    const unsigned int cmom,
    __global const double * restrict weights,
    __global const double * restrict scat_coef,

    __global const double * restrict angular0,
    __global const double * restrict angular1,
    __global const double * restrict angular2,
    __global const double * restrict angular3,
    __global const double * restrict angular4,
    __global const double * restrict angular5,
    __global const double * restrict angular6,
    __global const double * restrict angular7,

    __global const double * restrict angular_prev0,
    __global const double * restrict angular_prev1,
    __global const double * restrict angular_prev2,
    __global const double * restrict angular_prev3,
    __global const double * restrict angular_prev4,
    __global const double * restrict angular_prev5,
    __global const double * restrict angular_prev6,
    __global const double * restrict angular_prev7,

    __global const double * restrict time_delta,
    __global double * restrict scalar,
    __global double * restrict scalar_mom
    )
{
    // Cell index
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // For groups
    for (unsigned int g = 0; g < ng; g++)
    {
        double tot_g = 0.0;
        for (unsigned int l = 0; l < cmom-1; l++)
            scalar_mom(l,i,j,k,g) = 0.0;
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
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,0) + angular_prev(a,g,i,j,k,0)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,1) + angular_prev(a,g,i,j,k,1)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,2) + angular_prev(a,g,i,j,k,2)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,3) + angular_prev(a,g,i,j,k,3)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,4) + angular_prev(a,g,i,j,k,4)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,5) + angular_prev(a,g,i,j,k,5)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,6) + angular_prev(a,g,i,j,k,6)));
                tot_g += weights(a) * (0.5 * (angular(a,g,i,j,k,7) + angular_prev(a,g,i,j,k,7)));
                for (unsigned int l = 0; l < (cmom-1); l++)
                {
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,0) * weights(a) * (0.5 * (angular(a,g,i,j,k,0) + angular_prev(a,g,i,j,k,0)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,1) * weights(a) * (0.5 * (angular(a,g,i,j,k,1) + angular_prev(a,g,i,j,k,1)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,2) * weights(a) * (0.5 * (angular(a,g,i,j,k,2) + angular_prev(a,g,i,j,k,2)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,3) * weights(a) * (0.5 * (angular(a,g,i,j,k,3) + angular_prev(a,g,i,j,k,3)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,4) * weights(a) * (0.5 * (angular(a,g,i,j,k,4) + angular_prev(a,g,i,j,k,4)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,5) * weights(a) * (0.5 * (angular(a,g,i,j,k,5) + angular_prev(a,g,i,j,k,5)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,6) * weights(a) * (0.5 * (angular(a,g,i,j,k,6) + angular_prev(a,g,i,j,k,6)));
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,7) * weights(a) * (0.5 * (angular(a,g,i,j,k,7) + angular_prev(a,g,i,j,k,7)));
                }
            }
            else
            {
                tot_g += weights(a) * angular(a,g,i,j,k,0);
                tot_g += weights(a) * angular(a,g,i,j,k,1);
                tot_g += weights(a) * angular(a,g,i,j,k,2);
                tot_g += weights(a) * angular(a,g,i,j,k,3);
                tot_g += weights(a) * angular(a,g,i,j,k,4);
                tot_g += weights(a) * angular(a,g,i,j,k,5);
                tot_g += weights(a) * angular(a,g,i,j,k,6);
                tot_g += weights(a) * angular(a,g,i,j,k,7);
                for (unsigned int l = 0; l < cmom-1; l++)
                {
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,0) * weights(a) * angular(a,g,i,j,k,0);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,1) * weights(a) * angular(a,g,i,j,k,1);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,2) * weights(a) * angular(a,g,i,j,k,2);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,3) * weights(a) * angular(a,g,i,j,k,3);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,4) * weights(a) * angular(a,g,i,j,k,4);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,5) * weights(a) * angular(a,g,i,j,k,5);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,6) * weights(a) * angular(a,g,i,j,k,6);
                    scalar_mom(l,i,j,k,g) += scat_coef(a,l+1,7) * weights(a) * angular(a,g,i,j,k,7);
                }
            }
        }
        scalar(i,j,k,g) = tot_g;
    }
}


// Calculate the inverted denominator for all the energy groups
__kernel void calc_denominator(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,

    __global const double * restrict total_cross_section,
    __global const double * restrict time_delta,
    __global const double * restrict mu,
    const double dd_i,
    __global const double * restrict dd_j,
    __global const double * restrict dd_k,

    __global double * restrict denom
    )
{
    const unsigned int a_idx = get_global_id(0);
    const unsigned int g_idx = get_global_id(1);

    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < nx; i++)
            {
                denom(a_idx,g_idx,i,j,k) = 1.0 / (total_cross_section(g_idx,i,j,k) + time_delta(g_idx) + mu(a_idx)*dd_i + dd_j(a_idx) + dd_k(a_idx));
            }
        }
    }
}

// Calculate the time delta
__kernel void calc_time_delta(
    const double dt,
    __global const double * restrict velocity,
    __global double * restrict time_delta
    )
{
    const unsigned int g = get_global_id(0);
    time_delta(g) = 2.0 / (dt * velocity(g));
}


// Calculate the diamond difference coefficients
__kernel void calc_dd_coefficients(
    const double dy,
    const double dz,

    __global const double * restrict eta,
    __global const double * restrict xi,

    __global double * restrict dd_j,
    __global double * restrict dd_k
    )
{
    const unsigned int a = get_global_id(0);
    dd_j(a) = (2.0/dy)*eta(a);
    dd_k(a) = (2.0/dz)*xi(a);

}

// Calculate the total cross section from the spatial mapping
__kernel void calc_total_cross_section(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int nmat,
    __global const double * restrict xs,
    __global const unsigned int * restrict map,
    __global double * restrict total_cross_section
    )
{
    const unsigned int g = get_global_id(0);
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < nx; i++)
            {
                total_cross_section(g,i,j,k) = xs(map(i,j,k)-1,g);
            }
        }
    }
}

__kernel void calc_scattering_cross_section(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int nmom,
    const unsigned int nmat,
    __global const double * restrict gg_cs,
    __global const unsigned int * restrict map,
    __global double * restrict scat_cs
    )
{
    unsigned int g = get_global_id(0);

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                for (unsigned int l = 0; l < nmom; l++)
                    scat_cs(l,i,j,k,g) = gg_cs(map(i,j,k)-1,l,g,g);
}


// Calculate the outer source
__kernel void calc_outer_source(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int nmom,
    const unsigned int cmom,
    const unsigned int nmat,

    __global const int * restrict map,
    __global const double * restrict gg_cs,
    __global const double * restrict fixed_source,
    __global const int * restrict lma,

    __global const double * restrict scalar,
    __global const double * restrict scalar_mom,

    __global double * restrict g2g_source
    )
{
    const unsigned int g1 = get_global_id(0);

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
            {
                g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);
                for (unsigned int g2 = 0; g2 < ng; g2++)
                {
                    if (g1 == g2)
                        continue;

                    g2g_source(0,i,j,k,g1) += gg_cs(map(i,j,k)-1,0,g1,g2) * scalar(i,j,k,g2);

                    unsigned int mom = 1;
                    for (unsigned int l = 1; l < nmom; l++)
                    {
                        for (unsigned int m = 0; m < lma(l); m++)
                        {
                            g2g_source(mom,i,j,k,g1) += gg_cs(map(i,j,k)-1,l,g1,g2) * scalar_mom(mom-1,i,j,k,g2);
                            mom++;
                        }
                    }
                }
            }
}

// Calculate the inner source
__kernel void calc_inner_source(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int nmom,
    const unsigned int cmom,

    __global const double * restrict g2g_source,
    __global const double * restrict scat_cs,
    __global const double * restrict scalar,
    __global const double * restrict scalar_mom,
    __global const int * restrict lma,

    __global double * restrict source
    )
{
    const unsigned int g = get_global_id(0);

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
            {
                source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar(i,j,k,g);
                unsigned int mom = 1;
                for (unsigned int l = 1; l < nmom; l++)
                {
                    for (unsigned int m = 0; m < lma(l); m++)
                    {
                        source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) + scat_cs(l,i,j,k,g) * scalar_mom(mom-1,i,j,k,g);
                        mom++;
                    }
                }
            }
}

