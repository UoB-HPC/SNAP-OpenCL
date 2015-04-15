
#include "ocl_sweep.h"

// Set the global timestep variable to the current timestep
void ocl_set_timestep_(const unsigned int *timestep)
{
    global_timestep = (*timestep) - 1;
}

// Copy the source term to the OpenCL device
// source is the total source: qtot(cmom,nx,ny,nz,ng)
void copy_source_to_device_(double *source)
{
    cl_int err;
    err = clEnqueueWriteBuffer(queue[0], d_source, CL_TRUE, 0, sizeof(double)*cmom*nx*ny*nz*ng, source, 0, NULL, NULL);
    check_error(err, "Copying source buffer");
}

// Copy the angular flux update formula demoninator
void copy_denom_to_device_(double *denom)
{
    cl_int err;
    double *tmp = (double *)malloc(sizeof(double)*nang*ng*nx*ny*nz);
    // Transpose the denominator from the original SNAP format
    for (int a = 0; a < nang; a++)
        for (int g = 0; g < ng; g++)
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        tmp[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)] = denom[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*g)];

    err = clEnqueueWriteBuffer(queue[0], d_denom, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, tmp, 0, NULL, NULL);
    check_error(err, "Copying denom buffer");
    free(tmp);

}

void copy_dd_coefficients_to_device_(double *dd_i_, double *dd_j, double *dd_k)
{
    d_dd_i = *dd_i_;
    cl_int err;
    err = clEnqueueWriteBuffer(queue[0], d_dd_j, CL_TRUE, 0, sizeof(double)*nang, dd_j, 0, NULL, NULL);
    check_error(err, "Copying dd_j buffer");

    err = clEnqueueWriteBuffer(queue[0], d_dd_k, CL_TRUE, 0, sizeof(double)*nang, dd_k, 0, NULL, NULL);
    check_error(err, "Copying dd_k buffer");
}

void copy_time_delta_to_device_(double *time_delta)
{
    cl_int err;
    err = clEnqueueWriteBuffer(queue[0], d_time_delta, CL_TRUE, 0, sizeof(double)*ng, time_delta, 0, NULL, NULL);
    check_error(err, "Copying time_delta buffer");
}

void copy_total_cross_section_to_device_(double *total_cross_section)
{
    cl_int err;

    double *tmp = (double *)malloc(sizeof(double)*nx*ny*nz*ng);
    for (unsigned int i = 0; i < nx; i++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int k = 0; k < nz; k++)
                for (unsigned int g = 0; g < ng; g++)
                    tmp[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] = total_cross_section[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)];
    err = clEnqueueWriteBuffer(queue[0], d_total_cross_section, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
    check_error(err, "Copying total_cross_section buffer");
    free(tmp);
}

void zero_edge_flux_buffers_(void)
{
    cl_int err;

    size_t global[1] = {nang*ny*nz*ng};
    err = clSetKernelArg(k_zero_edge_array, 0, sizeof(cl_mem), &d_flux_i);
    check_error(err, "setting zero array flux_i");
    err = clEnqueueNDRangeKernel(queue[0], k_zero_edge_array, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue zero flux_i");

    global[0] = nang*nx*nz*ng;
    err = clSetKernelArg(k_zero_edge_array, 0, sizeof(cl_mem), &d_flux_j);
    check_error(err, "setting zero array flux_j");
    err = clEnqueueNDRangeKernel(queue[0], k_zero_edge_array, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue zero flux_j");

    global[0] = nang*nx*ny*ng;
    err = clSetKernelArg(k_zero_edge_array, 0, sizeof(cl_mem), &d_flux_k);
    check_error(err, "setting zero array flux_k");
    err = clEnqueueNDRangeKernel(queue[0], k_zero_edge_array, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue zero flux_k");
}

void zero_centre_flux_in_buffer_(void)
{
    cl_int err;
    double *zero = (double *)calloc(sizeof(double),nang*nx*ny*nz*ng);
    for (unsigned int o = 0; o < noct; o++)
    {
        err = clEnqueueWriteBuffer(queue[0], d_flux_in[o], CL_FALSE, 0, sizeof(double)*nang*nx*ny*nz*ng, zero, 0, NULL, NULL);
        check_error(err, "Copying flux_in to device");
    }
    err = clFinish(queue[0]);
    free(zero);
}

void zero_flux_moments_buffer(void)
{
    cl_int err;
    double *zero = calloc(sizeof(double),(cmom-1)*nx*ny*nz*ng);
    err = clEnqueueWriteBuffer(queue[0], d_scalar_mom, CL_TRUE, 0, sizeof(double)*(cmom-1)*nx*ny*nz*ng, zero, 0, NULL, NULL);
    check_error(err, "Zeroing scalar_mom buffer");
    free(zero);
}

void zero_scalar_flux(void)
{
    cl_int err;
    double *zero = calloc(sizeof(double),nx*ny*nz*ng);
    err = clEnqueueWriteBuffer(queue[0], d_scalar_flux, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, zero, 0, NULL, NULL);
    check_error(err, "Zeroing scalar_flux buffer");
    free(zero);
}

void zero_scalar_moments(void)
{
    cl_int err;
    double *zero = calloc(sizeof(double),(cmom-1)*nx*ny*nz*ng);
    err = clEnqueueWriteBuffer(queue[0], d_scalar_mom, CL_TRUE, 0, sizeof(double)*(cmom-1)*nx*ny*nz*ng, zero, 0, NULL, NULL);
    check_error(err, "Zeroing scalar_mom buffer");
    free(zero);
}

// Calculate denominator on the device
void calc_denom(void)
{
    cl_int err;
    const size_t global[2] = {nang, ng};

    err = clSetKernelArg(k_calc_denominator, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_calc_denominator, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_calc_denominator, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_calc_denominator, 3, sizeof(unsigned int), &nang);
    err |= clSetKernelArg(k_calc_denominator, 4, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_calc_denominator, 5, sizeof(cl_mem), &d_total_cross_section);
    err |= clSetKernelArg(k_calc_denominator, 6, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_calc_denominator, 7, sizeof(cl_mem), &d_mu);
    err |= clSetKernelArg(k_calc_denominator, 8, sizeof(double), &d_dd_i);
    err |= clSetKernelArg(k_calc_denominator, 9, sizeof(cl_mem), &d_dd_j);
    err |= clSetKernelArg(k_calc_denominator, 10, sizeof(cl_mem), &d_dd_k);
    err |= clSetKernelArg(k_calc_denominator, 11, sizeof(cl_mem), &d_denom);
    check_error(err, "Setting calc_denom arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_denominator, 2, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_denom kernel");
}

// Calculate diamond difference on the device
void calc_dd_coefficients(void)
{
    cl_int err;
    d_dd_i = 2.0 / dx;

    const size_t global[1] = {nang};
    err = clSetKernelArg(k_calc_dd_coefficients, 0, sizeof(double), &dy);
    err |= clSetKernelArg(k_calc_dd_coefficients, 1, sizeof(double), &dz);
    err |= clSetKernelArg(k_calc_dd_coefficients, 2, sizeof(cl_mem), &d_eta);
    err |= clSetKernelArg(k_calc_dd_coefficients, 3, sizeof(cl_mem), &d_xi);
    err |= clSetKernelArg(k_calc_dd_coefficients, 4, sizeof(cl_mem), &d_dd_j);
    err |= clSetKernelArg(k_calc_dd_coefficients, 5, sizeof(cl_mem), &d_dd_k);
    check_error(err, "Setting calc_dd_coefficients arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_dd_coefficients, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_dd_coefficients kernel");
}

// Calculate time delta on the device
void calc_time_delta(void)
{
    cl_int err;
    const size_t global[1] = {ng};

    err = clSetKernelArg(k_calc_time_delta, 0, sizeof(double), &dt);
    err |= clSetKernelArg(k_calc_time_delta, 1, sizeof(cl_mem), &d_velocity);
    err |= clSetKernelArg(k_calc_time_delta, 2, sizeof(cl_mem), &d_time_delta);
    check_error(err, "Setting calc_time_delta arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_time_delta, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_time_delta kernel");

}

// Calculate the total cross section on the device
void expand_cross_section(cl_mem * in, cl_mem * out)
{
    cl_int err;
    const size_t global[1] = {ng};

    err = clSetKernelArg(k_calc_total_cross_section, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_calc_total_cross_section, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_calc_total_cross_section, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_calc_total_cross_section, 3, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_calc_total_cross_section, 4, sizeof(unsigned int), &nmat);
    err |= clSetKernelArg(k_calc_total_cross_section, 5, sizeof(cl_mem), in);
    err |= clSetKernelArg(k_calc_total_cross_section, 6, sizeof(cl_mem), &d_map);
    err |= clSetKernelArg(k_calc_total_cross_section, 7, sizeof(cl_mem), out);
    check_error(err, "Setting calc_total_cross_section arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_total_cross_section, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_total_cross_section kernel");
}

void compute_outer_source(void)
{
    cl_int err;
    const size_t global[3] = {nx,ny,nz};

    err = clSetKernelArg(k_calc_outer_source, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_calc_outer_source, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_calc_outer_source, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_calc_outer_source, 3, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_calc_outer_source, 4, sizeof(unsigned int), &nmom);
    err |= clSetKernelArg(k_calc_outer_source, 5, sizeof(unsigned int), &cmom);
    err |= clSetKernelArg(k_calc_outer_source, 6, sizeof(unsigned int), &nmat);
    err |= clSetKernelArg(k_calc_outer_source, 7, sizeof(cl_mem), &d_map);
    err |= clSetKernelArg(k_calc_outer_source, 8, sizeof(cl_mem), &d_gg_cs);
    err |= clSetKernelArg(k_calc_outer_source, 9, sizeof(cl_mem), &d_fixed_source);
    err |= clSetKernelArg(k_calc_outer_source, 10, sizeof(cl_mem), &d_lma);
    err |= clSetKernelArg(k_calc_outer_source, 11, sizeof(cl_mem), &d_scalar_flux);
    err |= clSetKernelArg(k_calc_outer_source, 12, sizeof(cl_mem), &d_scalar_mom);
    err |= clSetKernelArg(k_calc_outer_source, 13, sizeof(cl_mem), &d_g2g_source);
    check_error(err, "Setting calc_outer_source arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_outer_source, 3, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_outer_source kernel");
}


void compute_inner_source(void)
{
    cl_int err;
    const size_t global[3] = {nx,ny,nz};

    err = clSetKernelArg(k_calc_inner_source, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_calc_inner_source, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_calc_inner_source, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_calc_inner_source, 3, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_calc_inner_source, 4, sizeof(unsigned int), &nmom);
    err |= clSetKernelArg(k_calc_inner_source, 5, sizeof(unsigned int), &cmom);
    err |= clSetKernelArg(k_calc_inner_source, 6, sizeof(cl_mem), &d_g2g_source);
    err |= clSetKernelArg(k_calc_inner_source, 7, sizeof(cl_mem), &d_scat_cs);
    err |= clSetKernelArg(k_calc_inner_source, 8, sizeof(cl_mem), &d_scalar_flux);
    err |= clSetKernelArg(k_calc_inner_source, 9, sizeof(cl_mem), &d_scalar_mom);
    err |= clSetKernelArg(k_calc_inner_source, 10, sizeof(cl_mem), &d_lma);
    err |= clSetKernelArg(k_calc_inner_source, 11, sizeof(cl_mem), &d_source);
    check_error(err, "Setting calc_inner_source arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_inner_source, 3, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_inner_source kernel");
}



void expand_scattering_cross_section(void)
{
    cl_int err;
    const size_t global[1] = {ng};

    err = clSetKernelArg(k_calc_scattering_cross_section, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 3, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 4, sizeof(unsigned int), &nmom);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 5, sizeof(unsigned int), &nmat);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 6, sizeof(cl_mem), &d_gg_cs);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 7, sizeof(cl_mem), &d_map);
    err |= clSetKernelArg(k_calc_scattering_cross_section, 8, sizeof(cl_mem), &d_scat_cs);
    check_error(err, "Setting calc_total_scattering_section arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_calc_scattering_cross_section, 1, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue calc_scattering_cross_section kernel");
}

bool check_convergence(double *old, double *new, double epsi)
{
    for (unsigned int g = 0; g < ng; g++)
        for (unsigned int k = 0; k < nz; k++)
            for (unsigned int j = 0; j < ny; j++)
                for (unsigned int i = 0; i < nx; i++)
                {
                    double val;
                    if (fabs(old[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] > tolr))
                    {
                        val = fabs(new[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)]/old[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] - 1.0);
                    }
                    else
                    {
                        val = fabs(new[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] - old[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)]);
                    }
                    if (val > epsi)
                    {
                        return false;
                    }
                }
    return true;
}



// Do the timestep, outer and inner iterations
void ocl_iterations_(void)
{
    cl_int err;
    double *old_outer_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
    double *new_outer_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
    double *old_inner_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
    double *new_inner_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
    bool outer_done;
    double t1 = omp_get_wtime();
    // Timestep loop
    for (unsigned int t = 0; t < timesteps; t++)
    {
        unsigned int tot_outers = 0;
        unsigned int tot_inners = 0;
        global_timestep = t;
        // Calculate data required at the beginning of each timestep
        zero_scalar_flux();
        zero_flux_moments_buffer();
        // Outer loop
        outer_done = false;
        for (unsigned int o = 0; o < outers; o++)
        {
            bool inner_done = false;
            tot_outers++;
            expand_cross_section(&d_xs, &d_total_cross_section);
            expand_scattering_cross_section();
            calc_dd_coefficients();
            calc_time_delta();
            calc_denom();
            // Compute the outer source
            compute_outer_source();
            // Save flux
            get_scalar_flux_(old_outer_scalar);
            // Inner loop
            for (unsigned int i = 0; i < inners; i++)
            {
                tot_inners++;
                // Compute the inner source
                compute_inner_source();
                // Save flux
                get_scalar_flux_(old_inner_scalar);
                zero_edge_flux_buffers_();
                // Sweep
                ocl_sweep_();
                // Scalar flux
                // ocl_scalar_flux_();
                reduce_angular_cells();
                reduce_moments_cells();
                // Check convergence
                get_scalar_flux_(new_inner_scalar);
                inner_done = check_convergence(old_inner_scalar, new_inner_scalar, epsi);
                if (inner_done)
                    break;
            }
            // Check convergence
            get_scalar_flux_(new_outer_scalar);
            outer_done = check_convergence(old_outer_scalar, new_outer_scalar, 100.0*epsi);
            if (outer_done && inner_done)
                break;
        }
        printf("Time %d -  %d outers, %d inners.\n", t, tot_outers, tot_inners);
        // Exit the time loop early if outer not converged
        if (!outer_done)
            break;
    }
    clFinish(queue[0]);
    double t2 = omp_get_wtime();

    printf("OpenCL: Time to convergence: %.3lfs\n", t2-t1);
    if (!outer_done)
        printf("Warning: did not converge\n");

    free(old_outer_scalar);
    free(new_outer_scalar);
    free(old_inner_scalar);
    free(new_inner_scalar);
}
