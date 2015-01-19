
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Include the kernel strings
#include "sweep_kernels.h"


// Global OpenCL handles (context, queue, etc.)
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

// OpenCL kernels
cl_kernel k_sweep_cell;

// OpenCL buffers
cl_mem d_source;
cl_mem d_flux_in;
cl_mem d_flux_out;
cl_mem d_denom;
cl_mem d_flux_halo_y;
cl_mem d_flux_halo_z;

// Check OpenCL errors and exit if no success
void check_error(cl_int err, char *msg)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error %d: %s\n", err, msg);
        exit(err);
    }
}

// Check for OpenCL build errors and display build messages
void check_build_error(cl_int err, char *msg)
{
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        char *build_log = (char*)malloc(sizeof(char)*2048);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*2048, build_log, NULL);
        fprintf(stderr, "Error: %d\n", err);
        fprintf(stderr, "Build log:\n%s\n", build_log);
        free(build_log);
    }
    check_error(err, msg);
}


void opencl_setup_(void)
{
    printf("Setting up OpenCL environment...");

    cl_int err;

    // Secure a platform
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    check_error(err, "Finding platforms");

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    check_error(err, "Getting platforms");

    // Get a device
    cl_device_type type = CL_DEVICE_TYPE_CPU;
    cl_platform_id platform;
    for (unsigned int i = 0; i < num_platforms; i++)
    {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], type, 0, NULL, &num_devices);
        if (num_devices > 0 && err == CL_SUCCESS)
        {
            err = clGetDeviceIDs(platforms[i], type, 1, &device, NULL);
            platform = platforms[i];
            check_error(err, "Securing a device");
            break;
        }
    }
    check_error(err, "Could not find a device");

    // Create a context
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "Creating command queue");

    // Create program
    program = clCreateProgramWithSource(context, 1, &sweep_kernels_ocl, NULL, &err);
    check_error(err, "Creating program");

    // Build program
    char *options = "";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    check_build_error(err, "Building program");

    // Create kernels
    k_sweep_cell = clCreateKernel(program, "sweep_cell", &err);
    check_error(err, "Creating kernel sweep_cell");

    free(platforms);
    printf("done\n");
}

// Release the global OpenCL handles
void opencl_teardown_(void)
{
    printf("Releasing OpenCL...");

    cl_int err;
    err = clReleaseDevice(device);
    check_error(err, "Releasing device");

    err = clReleaseContext(context);
    check_error(err, "Releasing context");

    err = clReleaseCommandQueue(queue);
    check_error(err, "Releasing queue");

    err = clReleaseMemObject(d_source);
    check_error(err, "Releasing source buffer");

    err = clReleaseMemObject(d_flux_in);
    check_error(err, "Releasing flux_in buffer");

    err = clReleaseMemObject(d_flux_out);
    check_error(err, "Releasing flux_out buffer");

    err = clReleaseMemObject(d_denom);
    check_error(err, "Releasing d_denom buffer");

    err = clReleaseMemObject(d_flux_halo_y);
    check_error(err, "Releasing d_flux_halo_y buffer");

    err = clReleaseMemObject(d_flux_halo_z);
    check_error(err, "Releasing d_flux_halo_z buffer");

    printf("done\n");
}


// Create buffers and copy the flux, source and
// cross section arrays to the OpenCL device
//
// Argument list:
// nx, ny, nz are the (local to MPI task) dimensions of the grid
// ng is the number of energy groups
// cmom is the "computational number of moments"
// ichunk is the number of yz planes in the KBA decomposition
// source is the total source: qtot(cmom,nx,ny,nz,ng)
// flux_in(nang,nx,ny,nz,noct,ng)   - Incoming time-edge flux pointer
// denom(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
// flux_halo_y: jb_in(nang,ichunk,nz,ng)  - y-dir boundary flux in from comm
// flux_halo_z: kb_in(nang,ichunk,ny,ng)  - z-dir boundary flux in from comm
int nx, ny, nz, ng, nang, noct, cmom, ichunk;
void copy_to_device_(
    int *nx_, int *ny_, int *nz_,
    int *ng_, int *nang_, int *noct_, int *cmom_,
    int *ichunk_,
    double *source, double *flux_in,
    double *denom,
    double *flux_halo_y, double *flux_halo_z)
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

    // Create buffers and copy data to device
    cl_int err;
    d_source = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*cmom*nx*ny*nz*ng, source, &err);
    check_error(err, "Creating source buffer");

    d_flux_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang*nx*ny*nz*noct*ng, flux_in, &err);
    check_error(err, "Creating flux_in buffer");

    d_flux_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*nang*nx*ny*nz*noct*ng, NULL, &err);
    check_error(err, "Creating flux_in buffer");

    d_denom = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang*nx*ny*nz*ng, denom, &err);
    check_error(err, "Creating denom buffer");

    d_flux_halo_y = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang*ichunk*nz*ng, flux_halo_y, &err);
    check_error(err, "Creating flux_halo_y buffer");

    d_flux_halo_z = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang*ichunk*ny*ng, flux_halo_z, &err);
    check_error(err, "Creating flux_halo_z buffer");

}

struct cell {
    unsigned int i,j,k;
};

typedef struct {
    unsigned int num_cells;
    struct cell *cells;
    // index is an index into the cells array for when storing the cell indexes
    unsigned int index;
} plane;

// Compute the order of the sweep for the first octant
plane *compute_sweep_order(void)
{
    unsigned int nplanes = ichunk + ny + nz - 2;
    plane *planes = (plane *)malloc(sizeof(plane)*nplanes);
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].num_cells = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                planes[n].num_cells++;
            }
        }
    }

    // Allocate the memory for each plane
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].cells = (struct cell *)malloc(sizeof(struct cell)*planes[i].num_cells);
        planes[i].index = 0;
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                unsigned int idx = planes[n].index;
                planes[n].cells[idx].i = i;
                planes[n].cells[idx].j = j;
                planes[n].cells[idx].k = k;
                planes[n].index += 1;
            }
        }
    }

    return planes;
}

// Enqueue the kernels to sweep over the grid and compute the angular flux
// Kernel: cell
// Work-group: energy group
// Work-item: angle
void sweep_octant_(void)
{
    cl_int err;

    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    // Get the order of cells to enqueue
    plane *planes = compute_sweep_order();

    const size_t global[] = {nang,ng};

    // Enqueue kernels
    for (int i = ndiag-1; i >= 0; i--)
    {
        for (unsigned int j = 0; j < planes[i].num_cells; j++)
        {
            err = clSetKernelArg(k_sweep_cell, 0, sizeof(unsigned int), &planes[i].cells[j].i);
            err |= clSetKernelArg(k_sweep_cell, 1, sizeof(unsigned int), &planes[i].cells[j].j);
            err |= clSetKernelArg(k_sweep_cell, 2, sizeof(unsigned int), &planes[i].cells[j].k);

            err |= clSetKernelArg(k_sweep_cell, 3, sizeof(int), &ichunk);
            err |= clSetKernelArg(k_sweep_cell, 4, sizeof(int), &nx);
            err |= clSetKernelArg(k_sweep_cell, 5, sizeof(int), &ny);
            err |= clSetKernelArg(k_sweep_cell, 6, sizeof(int), &nz);
            err |= clSetKernelArg(k_sweep_cell, 7, sizeof(int), &ng);
            err |= clSetKernelArg(k_sweep_cell, 8, sizeof(int), &nang);
            err |= clSetKernelArg(k_sweep_cell, 9, sizeof(int), &noct);


            err |= clSetKernelArg(k_sweep_cell, 10, sizeof(cl_mem), &d_flux_in);
            err |= clSetKernelArg(k_sweep_cell, 11, sizeof(cl_mem), &d_flux_out);
            err |= clSetKernelArg(k_sweep_cell, 12, sizeof(cl_mem), &d_source);
            err |= clSetKernelArg(k_sweep_cell, 13, sizeof(cl_mem), &d_denom);
            err |= clSetKernelArg(k_sweep_cell, 14, sizeof(cl_mem), &d_flux_halo_y);
            err |= clSetKernelArg(k_sweep_cell, 15, sizeof(cl_mem), &d_flux_halo_z);
            check_error(err, "Set sweep_cell kernel args");
            err = clEnqueueNDRangeKernel(queue, k_sweep_cell, 2, 0, global, NULL, 0, NULL, NULL);
            check_error(err, "Enqueue sweep_cell kernel");
            break;
        }
        break;
    }

    err = clFinish(queue);
    check_error(err, "Finish queue");

    // Free planes
    for (unsigned int i = 0; i < ndiag; i++)
    {
        free(planes[i].cells);
    }
    free(planes);
}

// Copy the flux_out buffer back to the host
void get_output_flux_(double* flux_out)
{
    cl_int err;
    err = clEnqueueReadBuffer(queue, d_flux_out, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, flux_out, 0, NULL, NULL);
    check_error(err, "Reading d_flux_out");
}
