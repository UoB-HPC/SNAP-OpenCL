
#include "ocl_sweep.h"

// Enqueue the kernel to reduce the angular flux to the scalar flux
void ocl_scalar_flux_(void)
{
    cl_int err;

    const size_t global[3] = {nx, ny, nz};

    err = clSetKernelArg(k_reduce_angular, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_reduce_angular, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_reduce_angular, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_reduce_angular, 3, sizeof(unsigned int), &nang);
    err |= clSetKernelArg(k_reduce_angular, 4, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_reduce_angular, 5, sizeof(unsigned int), &noct);
    err |= clSetKernelArg(k_reduce_angular, 6, sizeof(unsigned int), &cmom);

    err |= clSetKernelArg(k_reduce_angular, 7, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(k_reduce_angular, 8, sizeof(cl_mem), &d_scat_coeff);

    err |= clSetKernelArg(k_reduce_angular, 9, sizeof(cl_mem), &d_flux_out[0]);
    err |= clSetKernelArg(k_reduce_angular, 10, sizeof(cl_mem), &d_flux_out[1]);
    err |= clSetKernelArg(k_reduce_angular, 11, sizeof(cl_mem), &d_flux_out[2]);
    err |= clSetKernelArg(k_reduce_angular, 12, sizeof(cl_mem), &d_flux_out[3]);
    err |= clSetKernelArg(k_reduce_angular, 13, sizeof(cl_mem), &d_flux_out[4]);
    err |= clSetKernelArg(k_reduce_angular, 14, sizeof(cl_mem), &d_flux_out[5]);
    err |= clSetKernelArg(k_reduce_angular, 15, sizeof(cl_mem), &d_flux_out[6]);
    err |= clSetKernelArg(k_reduce_angular, 16, sizeof(cl_mem), &d_flux_out[7]);

    err |= clSetKernelArg(k_reduce_angular, 17, sizeof(cl_mem), &d_flux_in[0]);
    err |= clSetKernelArg(k_reduce_angular, 18, sizeof(cl_mem), &d_flux_in[1]);
    err |= clSetKernelArg(k_reduce_angular, 19, sizeof(cl_mem), &d_flux_in[2]);
    err |= clSetKernelArg(k_reduce_angular, 20, sizeof(cl_mem), &d_flux_in[3]);
    err |= clSetKernelArg(k_reduce_angular, 21, sizeof(cl_mem), &d_flux_in[4]);
    err |= clSetKernelArg(k_reduce_angular, 22, sizeof(cl_mem), &d_flux_in[5]);
    err |= clSetKernelArg(k_reduce_angular, 23, sizeof(cl_mem), &d_flux_in[6]);
    err |= clSetKernelArg(k_reduce_angular, 24, sizeof(cl_mem), &d_flux_in[7]);

    err |= clSetKernelArg(k_reduce_angular, 25, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_reduce_angular, 26, sizeof(cl_mem), &d_scalar_flux);
    err |= clSetKernelArg(k_reduce_angular, 27, sizeof(cl_mem), &d_scalar_mom);
    check_error(err, "Setting reduce_angular kernel arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_reduce_angular, 3, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue reduce_angular kernel");

    err = clFinish(queue[0]);
    check_error(err, "Finishing queue after reduce_angular kernel");

}
