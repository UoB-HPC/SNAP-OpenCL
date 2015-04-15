
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
#define scalar(g,i,j,k) scalar[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define weights(a) weights[(a)]

#define angular0(a,g,i,j,k) angular0[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular1(a,g,i,j,k) angular1[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular2(a,g,i,j,k) angular2[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular3(a,g,i,j,k) angular3[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular4(a,g,i,j,k) angular4[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular5(a,g,i,j,k) angular5[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular6(a,g,i,j,k) angular6[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular7(a,g,i,j,k) angular7[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define angular_prev0(a,g,i,j,k) angular_prev0[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev1(a,g,i,j,k) angular_prev1[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev2(a,g,i,j,k) angular_prev2[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev3(a,g,i,j,k) angular_prev3[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev4(a,g,i,j,k) angular_prev4[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev5(a,g,i,j,k) angular_prev5[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev6(a,g,i,j,k) angular_prev6[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev7(a,g,i,j,k) angular_prev7[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define velocity(g) velocity[(g)]

#define map(i,j,k) map[(i)+(nx*(j))+(nx*ny*(k))]
#define xs(i,g) xs[(i)+(nmat*(g))]

#define g2g_source(m,i,j,k,g) g2g_source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define fixed_source(i,j,k,g) fixed_source[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define gg_cs(m,l,g1,g2) gg_cs[(m)+(nmat*(l))+(nmat*nmom*(g1))+(nmat*nmom*ng*(g2))]
#define lma(m) lma[(m)]
#define scalar_mom(g,m,i,j,k) scalar_mom[(g)+((ng)*(m))+(ng*(cmom-1)*(i))+(ng*(cmom-1)*nx*(j))+(ng*(cmom-1)*nx*ny*(k))]

#define scat_cs(m,i,j,k,g) scat_cs[(m)+(nmom*(i))+(nmom*nx*(j))+(nmom*nx*ny*(k))+(nmom*nx*ny*nz*(g))]

struct cell {
    unsigned int i,j,k;
};

// Solve the transport equations for a single angle in a single cell for a single group
__kernel void sweep_cell(
    // Current cell index
    const int istep,
    const int jstep,
    const int kstep,
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
    __global const double * restrict denom,

    __global const struct cell * restrict cell_index
    )
{
    // Get indexes for angle and group
    const unsigned int a_idx = get_global_id(0) % nang;
    const unsigned int g_idx = get_global_id(0) / nang;
    const unsigned int i = (istep > 0) ? cell_index[get_global_id(1)].i : nx - cell_index[get_global_id(1)].i - 1;
    const unsigned int j = (jstep > 0) ? cell_index[get_global_id(1)].j : ny - cell_index[get_global_id(1)].j - 1;
    const unsigned int k = (kstep > 0) ? cell_index[get_global_id(1)].k : nz - cell_index[get_global_id(1)].k - 1;

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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int k = get_global_id(2);

    // For groups
    for (unsigned int g = 0; g < ng; g++)
    {
        double tot_g = 0.0;
        for (unsigned int l = 0; l < cmom-1; l++)
            scalar_mom(g,l,i,j,k) = 0.0;
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
                tot_g += weights(a) * (0.5 * (angular0(a,g,i,j,k) + angular_prev0(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular1(a,g,i,j,k) + angular_prev1(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular2(a,g,i,j,k) + angular_prev2(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular3(a,g,i,j,k) + angular_prev3(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular4(a,g,i,j,k) + angular_prev4(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular5(a,g,i,j,k) + angular_prev5(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular6(a,g,i,j,k) + angular_prev6(a,g,i,j,k)));
                tot_g += weights(a) * (0.5 * (angular7(a,g,i,j,k) + angular_prev7(a,g,i,j,k)));
                for (unsigned int l = 0; l < (cmom-1); l++)
                {
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,0) * weights(a) * (0.5 * (angular0(a,g,i,j,k) + angular_prev0(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,1) * weights(a) * (0.5 * (angular1(a,g,i,j,k) + angular_prev1(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,2) * weights(a) * (0.5 * (angular2(a,g,i,j,k) + angular_prev2(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,3) * weights(a) * (0.5 * (angular3(a,g,i,j,k) + angular_prev3(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,4) * weights(a) * (0.5 * (angular4(a,g,i,j,k) + angular_prev4(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,5) * weights(a) * (0.5 * (angular5(a,g,i,j,k) + angular_prev5(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,6) * weights(a) * (0.5 * (angular6(a,g,i,j,k) + angular_prev6(a,g,i,j,k)));
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,7) * weights(a) * (0.5 * (angular7(a,g,i,j,k) + angular_prev7(a,g,i,j,k)));
                }
            }
            else
            {
                tot_g += weights(a) * angular0(a,g,i,j,k);
                tot_g += weights(a) * angular1(a,g,i,j,k);
                tot_g += weights(a) * angular2(a,g,i,j,k);
                tot_g += weights(a) * angular3(a,g,i,j,k);
                tot_g += weights(a) * angular4(a,g,i,j,k);
                tot_g += weights(a) * angular5(a,g,i,j,k);
                tot_g += weights(a) * angular6(a,g,i,j,k);
                tot_g += weights(a) * angular7(a,g,i,j,k);
                for (unsigned int l = 0; l < (cmom-1); l++)
                {
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,0) * weights(a) * angular0(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,1) * weights(a) * angular1(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,2) * weights(a) * angular2(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,3) * weights(a) * angular3(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,4) * weights(a) * angular4(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,5) * weights(a) * angular5(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,6) * weights(a) * angular6(a,g,i,j,k);
                    scalar_mom(g,l,i,j,k) += scat_coef(a,l+1,7) * weights(a) * angular7(a,g,i,j,k);
                }
            }
        }
        scalar(g,i,j,k) = tot_g;
    }
}


// Reduce within each workgroup
// Requires each workgroup to be a power of two size
__kernel void reduce_angular_cell(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int noct,
    const unsigned int cmom,

    __local double * restrict scratch,

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
    __global double * restrict scalar
    )
{
    const unsigned int a = get_local_id(0);
    const unsigned int g = get_group_id(0);

    const double w = weights[a];

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
            {
                // Load into local memory
                scratch[a] = 0.0;
                if (a < nang)
                {
                    if (time_delta(g) != 0.0)
                    {
                        scratch[a] = w * (0.5 * (angular0(a,g,i,j,k) + angular_prev0(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular1(a,g,i,j,k) + angular_prev1(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular2(a,g,i,j,k) + angular_prev2(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular3(a,g,i,j,k) + angular_prev3(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular4(a,g,i,j,k) + angular_prev4(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular5(a,g,i,j,k) + angular_prev5(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular6(a,g,i,j,k) + angular_prev6(a,g,i,j,k)));
                        scratch[a] += w * (0.5 * (angular7(a,g,i,j,k) + angular_prev7(a,g,i,j,k)));
                    }
                    else
                    {
                        scratch[a] = w * angular0(a,g,i,j,k);
                        scratch[a] += w * angular1(a,g,i,j,k);
                        scratch[a] += w * angular2(a,g,i,j,k);
                        scratch[a] += w * angular3(a,g,i,j,k);
                        scratch[a] += w * angular4(a,g,i,j,k);
                        scratch[a] += w * angular5(a,g,i,j,k);
                        scratch[a] += w * angular6(a,g,i,j,k);
                        scratch[a] += w * angular7(a,g,i,j,k);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

                // Reduce in local memory
                for (unsigned int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
                {
                    if (a < offset)
                    {
                        scratch[a] += scratch[a + offset];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                // Save result
                if (a == 0)
                    scalar(g,i,j,k) = scratch[0];
            }
}

// Reduce the flux moments for a single cell
// One group per workgroup
// Requires workgroup size to be a power of two
__kernel void reduce_moments_cell(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int noct,
    const unsigned int cmom,

    __local double * restrict scratch,

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
    __global double * restrict scalar_mom
    )
{
    const unsigned int a = get_local_id(0);
    const unsigned int g = get_group_id(0);

    const double w = weights[a];

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                for (unsigned int l = 0; l < cmom-1; l++)
                {
                    // Load into local memory
                    scratch[a] = 0.0;
                    if (a < nang)
                    {
                        if (time_delta(g) != 0.0)
                        {
                            scratch[a] += scat_coef(a,l+1,0) * w * (0.5 * (angular0(a,g,i,j,k) + angular_prev0(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,1) * w * (0.5 * (angular1(a,g,i,j,k) + angular_prev1(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,2) * w * (0.5 * (angular2(a,g,i,j,k) + angular_prev2(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,3) * w * (0.5 * (angular3(a,g,i,j,k) + angular_prev3(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,4) * w * (0.5 * (angular4(a,g,i,j,k) + angular_prev4(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,5) * w * (0.5 * (angular5(a,g,i,j,k) + angular_prev5(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,6) * w * (0.5 * (angular6(a,g,i,j,k) + angular_prev6(a,g,i,j,k)));
                            scratch[a] += scat_coef(a,l+1,7) * w * (0.5 * (angular7(a,g,i,j,k) + angular_prev7(a,g,i,j,k)));
                        }
                        else
                        {
                            scratch[a] += scat_coef(a,l+1,0) * w * angular0(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,1) * w * angular1(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,2) * w * angular2(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,3) * w * angular3(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,4) * w * angular4(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,5) * w * angular5(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,6) * w * angular6(a,g,i,j,k);
                            scratch[a] += scat_coef(a,l+1,7) * w * angular7(a,g,i,j,k);
                        }
                    }

                    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

                    // Reduce in local memory
                    for (unsigned int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
                    {
                        if (a < offset)
                        {
                            scratch[a] += scratch[a + offset];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    // Save result
                    if (a == 0)
                        scalar_mom(g,l,i,j,k) = scratch[0];
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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int k = get_global_id(2);

    for (unsigned int g1 = 0; g1 < ng; g1++)
    {
        g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);
        for (unsigned int g2 = 0; g2 < ng; g2++)
        {
            if (g1 == g2)
                continue;

            g2g_source(0,i,j,k,g1) += gg_cs(map(i,j,k)-1,0,g2,g1) * scalar(g2,i,j,k);

            unsigned int mom = 1;
            for (unsigned int l = 1; l < nmom; l++)
            {
                for (int m = 0; m < lma(l); m++)
                {
                    g2g_source(mom,i,j,k,g1) += gg_cs(map(i,j,k)-1,l,g2,g1) * scalar_mom(g2,mom-1,i,j,k);
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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int k = get_global_id(2);

    for (unsigned int g = 0; g < ng; g++)
    {
        source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar(g,i,j,k);
        unsigned int mom = 1;
        for (unsigned int l = 1; l < nmom; l++)
        {
            for (int m = 0; m < lma(l); m++)
            {
                source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) + scat_cs(l,i,j,k,g) * scalar_mom(g,mom-1,i,j,k);
                mom++;
            }
        }
    }
}

// Zero an edge array
__kernel void zero_edge_array(__global double * array)
{
    const unsigned int i = get_global_id(0);
    array[i] = 0.0;
}
