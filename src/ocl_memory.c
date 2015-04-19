
#include "ocl_sweep.h"

// Forward declare to zero buffer functions
extern void zero_edge_flux_buffers_(void);
extern void zero_centre_flux_in_buffer_(void);
extern void zero_centre_flux_out_buffer_(void);
extern void zero_scalar_flux(void);
extern void zero_scalar_moments(void);

// Check the devices available memory to check everything will fit in the device
void check_device_memory(void)
{
    cl_int err;
    cl_ulong max_alloc;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, NULL);
    check_error(err, "Getting max allocation size");
    unsigned int memory = sizeof(double)*nang*ng*nx*ny*nz;
    if (max_alloc < memory)
    {
        fprintf(stderr, "Error: Device does not support a big enough array for the angular flux\n");
        exit(-1);
    }

    cl_ulong global_memory;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_memory, NULL);
    check_error(err, "Getting global memory size");
    if (global_memory < 2 * noct * memory)
    {
        fprintf(stderr, "Error: Device does not have enough global memory for angular flux arrays\n");
        fprintf(stderr, "Required: %.1f GB\n", (2.0 * noct * memory)/(1024.0*1024.0*1024.0));
        fprintf(stderr, "Available %.1f GB.\n", (float)global_memory/(1024.0*1024.0*1024.0));
        exit(-1);
    }

    // Calculate total memory usage
    unsigned long total = 0;
    total += sizeof(double)*nang*nx*ny*nz*ng*noct;
    total += sizeof(double)*nang*nx*ny*nz*ng*noct;
    total += sizeof(double)*nang*ny*nz*ng;
    total += sizeof(double)*nang*nx*nz*ng;
    total += sizeof(double)*nang*nx*ny*ng;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang*cmom*noct;
    total += sizeof(double)*nx*ny*nz*ng;
    total += sizeof(double)*nmat*ng;
    total += sizeof(int)*nx*ny*nz;
    total += sizeof(double)*nx*ny*nz*ng;
    total += sizeof(double)*nmat*nmom*ng*ng;
    total += sizeof(double)*nmom;
    total += sizeof(double)*cmom*nx*ny*nz*ng;
    if (cmom == 1)
        total += sizeof(double)*nx*ny*nz*ng;
    else
        total += sizeof(double)*(cmom-1)*nx*ny*nz*ng;
    total += sizeof(double)*nmom*nx*ny*nz*ng;
    total += sizeof(double)*nang;
    total += sizeof(double)*nang*nx*ny*nz*ng;
    total += sizeof(double)*cmom*nx*ny*nz*ng;
    total += sizeof(double)*ng;
    total += sizeof(double)*ng;
    total += sizeof(double)*nx*ny*nz*ng;

    if (global_memory < total)
    {
        fprintf(stderr, "Error: Device does not have enough global memory for all the buffers\n");
        fprintf(stderr, "Required: %.1f GB\n", (double)total/(1024.0*1024.0*1024.0));
        fprintf(stderr, "Available %.1f GB.\n", (double)global_memory/(1024.0*1024.0*1024.0));
        exit(-1);
    }
}

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
    double *mu, double *eta, double *xi,
    double *scat_coef,
    double *total_cross_section,
    double *weights,
    double *velocity,
    double *xs,
    int *mat,
    double *fixed_source,
    double *gg_cs,
    int *lma,
    double *g2g_source,
    double *flux_in)
{

    check_device_memory();

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

    d_flux_in = malloc(sizeof(cl_mem)*noct);
    d_flux_out = malloc(sizeof(cl_mem)*noct);

    for (unsigned int o = 0; o < noct; o++)
    {
        d_flux_in[o] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*nz*ng, NULL, &err);
        check_error(err, "Creating flux_in buffer");

        d_flux_out[o] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*nz*ng, NULL, &err);
        check_error(err, "Creating flux_out buffer");
    }

    zero_centre_flux_in_buffer_();
    zero_centre_flux_out_buffer_();

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

    d_dd_j = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating dd_j buffer");

    d_dd_k = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating dd_k buffer");

    d_mu = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating mu buffer");
    err = clEnqueueWriteBuffer(queue[0], d_mu, CL_FALSE, 0, sizeof(double)*nang, mu, 0, NULL, NULL);
    check_error(err, "Copying mu buffer");

    d_eta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang, eta, &err);
    check_error(err, "Creating eta buffer");

    d_xi = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang, xi, &err);
    check_error(err, "Creating xi buffer");

    d_scat_coeff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang*cmom*noct, NULL, &err);
    check_error(err, "Creating scat_coef buffer");
    err = clEnqueueWriteBuffer(queue[0], d_scat_coeff, CL_FALSE, 0, sizeof(double)*nang*cmom*noct, scat_coef, 0, NULL, NULL);
    check_error(err, "Copying scat_coef buffer");


    d_total_cross_section = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating total_cross_section buffer");

    d_xs = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*nmat*ng, xs, &err);
    check_error(err, "Creating xs buffer");

    d_map = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*nx*ny*nz, NULL, &err);
    check_error(err, "Creating map buffer");
    err = clEnqueueWriteBuffer(queue[0], d_map, CL_TRUE, 0, sizeof(int)*nx*ny*nz, mat, 0, NULL, NULL);
    check_error(err, "Copying mat buffer");

    d_fixed_source = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nx*ny*nz*ng, fixed_source, &err);
    check_error(err, "Creating fixed_source buffer");

    d_gg_cs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nmat*nmom*ng*ng, gg_cs, &err);
    check_error(err, "Creating gg_cs buffer");

    d_lma = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nmom, lma, &err);
    check_error(err, "Creating lma buffer");

    d_g2g_source = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*cmom*nx*ny*nz*ng, g2g_source, &err);
    check_error(err, "Creating g2g_source buffer");

    if (cmom == 1)
        d_scalar_mom = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    else
        d_scalar_mom = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(cmom-1)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating scalar_mom buffer");
    zero_scalar_moments();

    d_scat_cs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nmom*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating scat_cs buffer");

    d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang, NULL, &err);
    check_error(err, "Creating weights buffer");
    err = clEnqueueWriteBuffer(queue[0], d_weights, CL_FALSE, 0, sizeof(double)*nang, weights, 0, NULL, NULL);
    check_error(err, "Copying weights buffer");

    // Create buffers written to later
    d_denom = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nang*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating denom buffer");

    d_source = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*cmom*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating source buffer");

    d_time_delta = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*ng, NULL, &err);
    check_error(err, "Creating time_delta buffer");

    d_velocity = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*ng, velocity, &err);
    check_error(err, "Creating velocity buffer");

    d_scalar_flux = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating scalar_flux buffer");
    zero_scalar_flux();

    // Wait for the data to be on the device before returning
    err = clFinish(queue[0]);
    check_error(err, "Waiting for queue after buffer init");
}

// Copy the scalar flux value back to the host
void get_scalar_flux_(double *scalar, bool block)
{
    cl_bool blocking = (block) ? CL_TRUE : CL_FALSE;
    cl_int err;
    err = clEnqueueReadBuffer(queue[0], d_scalar_flux, blocking, 0, sizeof(double)*nx*ny*nz*ng, scalar, 0, NULL, NULL);
    check_error(err, "Enqueue read scalar_flux buffer");
}

// Copy the scalar flux value back to the host and transpose
void get_scalar_flux_trans_(double *scalar)
{
    double *tmp = malloc(sizeof(double)*nx*ny*nz*ng);
    cl_int err;
    err = clEnqueueReadBuffer(queue[0], d_scalar_flux, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
    check_error(err, "Enqueue read scalar_flux buffer");
    for (unsigned int g = 0; g < ng; g++)
        for (unsigned int i = 0; i < nx; i++)
            for (unsigned int j = 0; j < ny; j++)
                for (unsigned int k = 0; k < nz; k++)
                    scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] = tmp[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)];

    free(tmp);
}

void get_scalar_flux_moments_(double *scalar_moments)
{
    double *tmp = malloc(sizeof(double)*(cmom-1)*nx*ny*nz*ng);
    cl_int err;
    err = clEnqueueReadBuffer(queue[0], d_scalar_mom, CL_TRUE, 0, sizeof(double)*(cmom-1)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
    check_error(err, "Enqueue read scalar_mom buffer");
    for (unsigned int g = 0; g < ng; g++)
        for (unsigned int l = 0; l < cmom-1; l++)
            for (unsigned int i = 0; i < nx; i++)
                for (unsigned int j = 0; j < ny; j++)
                    for (unsigned int k = 0; k < nz; k++)
                        scalar_moments[l+((cmom-1)*i)+((cmom-1)*nx*j)+((cmom-1)*nx*ny*k)+((cmom-1)*nx*ny*nz*g)] = tmp[g+(ng*l)+(ng*(cmom-1)*i)+(ng*(cmom-1)*nx*j)+(ng*(cmom-1)*nx*ny*k)];
    free(tmp);
}


// Copy the flux_out buffer back to the host
void get_output_flux_(double* flux_out)
{
    double *tmp = calloc(sizeof(double),nang*ng*nx*ny*nz*noct);
    cl_int err;
    for (unsigned int o = 0; o < noct; o++)
    {
        if (global_timestep % 2 == 0)
            err = clEnqueueReadBuffer(queue[0], d_flux_out[o], CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, &(tmp[nang*ng*nx*ny*nz*o]), 0, NULL, NULL);
        else
            err = clEnqueueReadBuffer(queue[0], d_flux_in[o], CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, &(tmp[nang*ng*nx*ny*nz*o]), 0, NULL, NULL);
    }
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
