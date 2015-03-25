
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
    err = clEnqueueWriteBuffer(queue[0], d_flux_in, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, zero, 0, NULL, NULL);
    free(zero);
    check_error(err, "Copying flux_in to device");
}