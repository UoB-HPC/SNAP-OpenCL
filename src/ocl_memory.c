
#include "ocl_sweep.h"

// Forward declare to zero buffer functions
extern void zero_edge_flux_buffers_(void);
extern void zero_centre_flux_in_buffer_(void);

// Create buffers and copy the flux, source and
// cross section arrays to the OpenCL device
//
// Argument list:
// nx, ny, nz are the (local to MPI task) dimensions of the grid
// ng is the number of energy groups
// cmom is the "computational number of moments"
// ichunk is the number of yz planes in the KBA decomposition
// dd_i, dd_j(nang), dd_k(nang) is the x,y,z (resp) diamond difference coefficients
// mu(nang) is x-direction cosines
// scat_coef [ec](nang,cmom,noct) - Scattering expansion coefficients
// time_delta [vdelt](ng)              - time-absorption coefficient
// total_cross_section [t_xs](nx,ny,nz,ng)       - Total cross section on mesh
// flux_in(nang,nx,ny,nz,noct,ng)   - Incoming time-edge flux pointer
// denom(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
// weights(nang) - angle weights for scalar reduction
void copy_to_device_(
    int *nx_, int *ny_, int *nz_,
    int *ng_, int *nang_, int *noct_, int *cmom_,
    int *ichunk_,
    double *mu, double *scat_coef,
    double *total_cross_section,
    double *weights,
    double *flux_in)
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

    // Create array for OpenCL events - one for each cell
    events = calloc(sizeof(cl_event),nx*ny*nz);

    // Create zero array for the edge flux buffers
    // First we need maximum two of nx, ny and nz
    size_t s = nang * ng;
    if (nx < ny && nx < nz)
        s *= ny * nz;
    else if (ny < nx && ny < nz)
        s *= nx * nz;
    else
        s *= nx * ny;
    zero_edge = (double *)calloc(sizeof(double), s);


    // Create buffers and copy data to device
    cl_int err;

    d_flux_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*nz*noct*ng, NULL, &err);
    check_error(err, "Creating flux_in buffer");

    d_flux_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*nz*noct*ng, NULL, &err);
    check_error(err, "Creating flux_out buffer");

    zero_centre_flux_in_buffer_();

    // flux_i(nang,ny,nz,ng)     - Working psi_x array (edge pointers)
    // flux_j(nang,ichunk,nz,ng) - Working psi_y array
    // flux_k(nang,ichunk,ny,ng) - Working psi_z array

    d_flux_i = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*ny*nz*ng, NULL, &err);
    check_error(err, "Creating flux_i buffer");

    d_flux_j = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*nz*ng, NULL, &err);
    check_error(err, "Creating flux_j buffer");

    d_flux_k = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*ng, NULL, &err);
    check_error(err, "Creating flux_k buffer");

    zero_edge_flux_buffers_();

    d_dd_j = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating dd_j buffer");

    d_dd_k = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating dd_k buffer");

    d_mu = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating mu buffer");
    err = clEnqueueWriteBuffer(queue[0], d_mu, CL_FALSE, 0, sizeof(double)*nang, mu, 0, NULL, NULL);
    check_error(err, "Copying mu buffer");

    d_scat_coeff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang*cmom*noct, NULL, &err);
    check_error(err, "Creating scat_coef buffer");
    err = clEnqueueWriteBuffer(queue[0], d_scat_coeff, CL_FALSE, 0, sizeof(double)*nang*cmom*noct, scat_coef, 0, NULL, NULL);
    check_error(err, "Copying scat_coef buffer");


    d_total_cross_section = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating total_cross_section buffer");

    // Reorder the memory layout before copy
    double *tmp = (double *)malloc(sizeof(double)*nx*ny*nz*ng);
    for (unsigned int i = 0; i < nx; i++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int k = 0; k < nz; k++)
                for (unsigned int g = 0; g < ng; g++)
                    tmp[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] = total_cross_section[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)];
    err = clEnqueueWriteBuffer(queue[0], d_total_cross_section, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
    check_error(err, "Copying total_cross_section buffer");
    free(tmp);

    d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating weights buffer");
    err = clEnqueueWriteBuffer(queue[0], d_weights, CL_FALSE, 0, sizeof(double)*nang, weights, 0, NULL, NULL);
    check_error(err, "Copying weights buffer");

    // Create buffers written to later
    d_denom = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating denom buffer");

    d_source = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*cmom*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating source buffer");

    d_time_delta = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*ng, NULL, &err);
    check_error(err, "Creating time_delta buffer");

    d_scalar_flux = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating scalar_flux buffer");

    // Wait for the data to be on the device before returning
    err = clFinish(queue[0]);
    check_error(err, "Waiting for queue after buffer init");
}

// Copy the scalar flux value back to the host
void get_scalar_flux_(double *scalar)
{
    cl_int err;
    err = clEnqueueReadBuffer(queue[0], d_scalar_flux, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, scalar, 0, NULL, NULL);
    check_error(err, "Enqueue read scalar_flux buffer");
}


// Copy the flux_out buffer back to the host
void get_output_flux_(double* flux_out)
{
    double *tmp = calloc(sizeof(double),nang*ng*nx*ny*nz*noct);
    cl_int err;
    if (global_timestep % 2 == 0)
        err = clEnqueueReadBuffer(queue[0], d_flux_out, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, tmp, 0, NULL, NULL);
    else
        err = clEnqueueReadBuffer(queue[0], d_flux_in, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, tmp, 0, NULL, NULL);
    check_error(err, "Reading d_flux_out");

    // Transpose the data into the original SNAP format
    for (int a = 0; a < nang; a++)
        for (int g = 0; g < ng; g++)
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        for (int o = 0; o < noct; o++)
                            flux_out[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)] = tmp[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)];
    free(tmp);
}