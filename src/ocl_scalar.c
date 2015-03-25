
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

    err |= clSetKernelArg(k_reduce_angular, 6, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(k_reduce_angular, 7, sizeof(cl_mem), &d_flux_out);
    err |= clSetKernelArg(k_reduce_angular, 8, sizeof(cl_mem), &d_flux_in);
    err |= clSetKernelArg(k_reduce_angular, 9, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_reduce_angular, 10, sizeof(cl_mem), &d_scalar_flux);
    check_error(err, "Setting reduce_angular kernel arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_reduce_angular, 3, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue reduce_angular kernel");

    err = clFinish(queue[0]);
    check_error(err, "Finishing queue after reduce_angular kernel");

}
