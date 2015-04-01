
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
    err = clEnqueueWriteBuffer(queue[0], d_flux_i, CL_FALSE, 0, sizeof(double)*nang*ny*nz*ng, zero_edge, 0, NULL, NULL);
    check_error(err, "Zeroing flux_i buffer");

    err = clEnqueueWriteBuffer(queue[0], d_flux_j, CL_FALSE, 0, sizeof(double)*nang*nx*nz*ng, zero_edge, 0, NULL, NULL);
    check_error(err, "Zeroing flux_j buffer");

    err = clEnqueueWriteBuffer(queue[0], d_flux_k, CL_FALSE, 0, sizeof(double)*nang*nx*ny*ng, zero_edge, 0, NULL, NULL);
    check_error(err, "Zeroing flux_k buffer");
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
    clFinish(queue[0]);

    double * tmp = malloc(sizeof(double)*nang*ng*nx*ny*nz);
    clEnqueueReadBuffer(queue[0], d_denom, CL_TRUE, 0, sizeof(double)*nang*ng*nx*ny*nz, tmp, 0, NULL, NULL);
    free(tmp);
}

// Calculate diamond difference on the device
void calc_dd_coefficients(void)
{
    // Todo
}

// Calculate time delta on the device
void calc_time_delta(void)
{
    // Todo
}

// Do the timestep, outer and inner iterations
void ocl_iterations_(void)
{
    // Timestep loop
    for (unsigned int t = 0; t < timesteps; t++)
    {
        calc_dd_coefficients();
        calc_time_delta();
        calc_denom();
        // Outer loop
        for (unsigned int o = 0; o < outers; o++)
        {
            // Inner loop
            for (unsigned int i = 0; i < inners; i++)
            {
                ;
            }
        }
    }
    clFinish(queue[0]);

}
