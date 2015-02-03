
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <omp.h>

// Include the kernel strings
#include "sweep_kernels.h"

#define NUM_QUEUES 4

// Global OpenCL handles (context, queue, etc.)
cl_device_id device;
cl_context context;
cl_command_queue queue[NUM_QUEUES];
cl_program program;

// OpenCL kernels
cl_kernel k_sweep_cell;
cl_kernel k_reduce_angular;

// OpenCL buffers
cl_mem d_source;
cl_mem d_flux_in;
cl_mem d_flux_out;
cl_mem d_flux_i;
cl_mem d_flux_j;
cl_mem d_flux_k;
cl_mem d_denom;
cl_mem d_dd_j;
cl_mem d_dd_k;
cl_mem d_mu;
cl_mem d_scat_coeff;
cl_mem d_time_delta;
cl_mem d_total_cross_section;
cl_mem d_weights;
cl_mem d_scalar_flux;

// Global variable for the timestep
unsigned int global_timestep;

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
        char *build_log = (char*)malloc(sizeof(char)*4096);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*4096, build_log, NULL);
        fprintf(stderr, "Error: %d\n", err);
        fprintf(stderr, "Build log:\n%s\n", build_log);
        free(build_log);
    }
    check_error(err, msg);
}


#define MAX_DEVICES 12
#define MAX_INFO_STRING 128

unsigned int get_devices(cl_device_id devices[MAX_DEVICES])
{
    cl_int err;
    // Get platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    check_error(err, "Finding platforms");

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    check_error(err, "Getting platforms");

    // Get all devices
    cl_uint num_devices = 0;
    for (unsigned int i = 0; i < num_platforms; i++)
    {
        cl_uint num;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES-num_devices, devices+num_devices, &num);
        check_error(err, "Getting devices");
        num_devices += num;
    }
    return num_devices;
}

void list_devices(void)
{
    cl_device_id devices[MAX_DEVICES];
    unsigned int num_devices = get_devices(devices);
    printf("\nFound %d devices:\n", num_devices);
    for (unsigned int i = 0; i < num_devices; i++)
    {
        char name[MAX_INFO_STRING];
        cl_int err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
        check_error(err, "Getting device name");
        printf("%d: %s\n", i, name);
    }
}

void opencl_setup_(void)
{
    printf("Setting up OpenCL environment...\n\n");

    cl_int err;

    // Use the first device by default
    int device_index = 0;

    // Check for the OpenCL config file stored in SNAP_OCL env variable
    char *device_string = getenv("SNAP_OCL_DEVICE");
    if (device_string != NULL)
    {
        device_index = strtol(device_string, NULL, 10);
        if (device_index ==  -1)
        {
            // List devices and then quit
            list_devices();
            exit(1);
        }
    }

    // Get the first or chosen device
    cl_device_id devices[MAX_DEVICES];
    unsigned int num_devices = get_devices(devices);
    cl_device_id device = devices[device_index];

    // Print device name
    char name[MAX_INFO_STRING];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
    check_error(err, "Getting device name");
    printf("Running on %s\n", name);

    // Create a context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating context");

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; i++)
    {
        queue[i] = clCreateCommandQueue(context, device, 0, &err);
        check_error(err, "Creating command queue");
    }

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

    k_reduce_angular = clCreateKernel(program, "reduce_angular", &err);
    check_error(err, "Creating kernel reduce_angular");

    printf("\nOpenCL environment setup complete\n\n");

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

    for (int i = 0; i < NUM_QUEUES; i++)
    {
        err = clReleaseCommandQueue(queue[i]);
        check_error(err, "Releasing queue");
    }

    err = clReleaseMemObject(d_source);
    check_error(err, "Releasing source buffer");

    err = clReleaseMemObject(d_flux_in);
    check_error(err, "Releasing flux_in buffer");

    err = clReleaseMemObject(d_flux_out);
    check_error(err, "Releasing flux_out buffer");

    err = clReleaseMemObject(d_denom);
    check_error(err, "Releasing d_denom buffer");

    printf("done\n");
}


// Forward declare to zero buffer functions
void zero_edge_flux_buffers_(void);
void zero_centre_flux_in_buffer_(void);



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
int nx, ny, nz, ng, nang, noct, cmom, ichunk;
double d_dd_i;
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

    d_mu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang, mu, &err);
    check_error(err, "Creating mu buffer");

    d_scat_coeff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang*cmom*noct, scat_coef, &err);
    check_error(err, "Creating scat_coef buffer");

    d_total_cross_section = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nx*ny*nz*ng, total_cross_section, &err);
    check_error(err, "Creating total_cross_section buffer");

    d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*nang, weights, &err);
    check_error(err, "Creating weights buffer");

    // Create buffers written to later
    d_denom = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*nang*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating denom buffer");

    d_source = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*cmom*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating source buffer");

    d_time_delta = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*ng, NULL, &err);
    check_error(err, "Creating time_delta buffer");

    d_scalar_flux = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*nx*ny*nz*ng, NULL, &err);
    check_error(err, "Creating scalar_flux buffer");

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
    err = clEnqueueWriteBuffer(queue[0], d_denom, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, denom, 0, NULL, NULL);
    check_error(err, "Copying denom buffer");

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
    double *zero = (double *)calloc(sizeof(double),nang*ny*nz*ng);
    err = clEnqueueWriteBuffer(queue[0], d_flux_i, CL_TRUE, 0, sizeof(double)*nang*ny*nz*ng, zero, 0, NULL, NULL);
    free(zero);
    check_error(err, "Zeroing flux_i buffer");

    zero = (double *)calloc(sizeof(double),nang*nx*nz*ng);
    err = clEnqueueWriteBuffer(queue[0], d_flux_j, CL_TRUE, 0, sizeof(double)*nang*nx*nz*ng, zero, 0, NULL, NULL);
    free(zero);
    check_error(err, "Zeroing flux_j buffer");

    zero = (double *)calloc(sizeof(double),nang*nx*ny*ng);
    err = clEnqueueWriteBuffer(queue[0], d_flux_k, CL_TRUE, 0, sizeof(double)*nang*nx*ny*ng, zero, 0, NULL, NULL);
    free(zero);
    check_error(err, "Zeroing flux_k buffer");
}

void zero_centre_flux_in_buffer_(void)
{
    cl_int err;
    double *zero = (double *)calloc(sizeof(double),nang*nx*ny*nz*noct*ng);
    err = clEnqueueWriteBuffer(queue[0], d_flux_in, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, zero, 0, NULL, NULL);
    free(zero);
    check_error(err, "Copying flux_in to device");
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
void enqueue_octant(const unsigned int timestep, const unsigned int oct, const unsigned int ndiag, const plane *planes)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    int order_3d = 0b000001010011100101110111;

    int order_2d = 0b11100100;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 0;
    int jstep = (yhi == ny) ? -1 : 0;
    int kstep = (zhi == nz) ? -1 : 0;


    cl_int err;

    const size_t global[2] = {nang, ng};

    // Set the kernel arguments
    err = clSetKernelArg(k_sweep_cell, 3, sizeof(unsigned int), &oct);
    err |= clSetKernelArg(k_sweep_cell, 4, sizeof(int), &ichunk);
    err |= clSetKernelArg(k_sweep_cell, 5, sizeof(int), &nx);
    err |= clSetKernelArg(k_sweep_cell, 6, sizeof(int), &ny);
    err |= clSetKernelArg(k_sweep_cell, 7, sizeof(int), &nz);
    err |= clSetKernelArg(k_sweep_cell, 8, sizeof(int), &ng);
    err |= clSetKernelArg(k_sweep_cell, 9, sizeof(int), &nang);
    err |= clSetKernelArg(k_sweep_cell, 10, sizeof(int), &noct);
    err |= clSetKernelArg(k_sweep_cell, 11, sizeof(int), &cmom);

    err |= clSetKernelArg(k_sweep_cell, 12, sizeof(double), &d_dd_i);
    err |= clSetKernelArg(k_sweep_cell, 13, sizeof(cl_mem), &d_dd_j);
    err |= clSetKernelArg(k_sweep_cell, 14, sizeof(cl_mem), &d_dd_k);
    err |= clSetKernelArg(k_sweep_cell, 15, sizeof(cl_mem), &d_mu);
    err |= clSetKernelArg(k_sweep_cell, 16, sizeof(cl_mem), &d_scat_coeff);
    err |= clSetKernelArg(k_sweep_cell, 17, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_sweep_cell, 18, sizeof(cl_mem), &d_total_cross_section);

    // Swap the angular flux pointers if necessary
    // Even timesteps: Read flux_in and write to flux_out
    // Odd timesteps: Read flux_out and write to flux_in
    if (timestep % 2 == 0)
    {
        err |= clSetKernelArg(k_sweep_cell, 19, sizeof(cl_mem), &d_flux_in);
        err |= clSetKernelArg(k_sweep_cell, 20, sizeof(cl_mem), &d_flux_out);
    }
    else
    {
        err |= clSetKernelArg(k_sweep_cell, 19, sizeof(cl_mem), &d_flux_out);
        err |= clSetKernelArg(k_sweep_cell, 20, sizeof(cl_mem), &d_flux_in);
    }

    err |= clSetKernelArg(k_sweep_cell, 21, sizeof(cl_mem), &d_flux_i);
    err |= clSetKernelArg(k_sweep_cell, 22, sizeof(cl_mem), &d_flux_j);
    err |= clSetKernelArg(k_sweep_cell, 23, sizeof(cl_mem), &d_flux_k);


    err |= clSetKernelArg(k_sweep_cell, 24, sizeof(cl_mem), &d_source);
    err |= clSetKernelArg(k_sweep_cell, 25, sizeof(cl_mem), &d_denom);
    check_error(err, "Set sweep_cell kernel args");

    // Loop over the diagonal wavefronts
    for (unsigned int d = 0; d < ndiag; d++)
    {
        // Loop through the list of cells in this plane
        for (unsigned int l = 0; l < planes[d].num_cells; l++)
        {
            // Calculate the real cell index for this octant
            int i = planes[d].cells[l].i;
            int j = planes[d].cells[l].j;
            int k = planes[d].cells[l].k;
            if (istep < 0)
                i = nx - i - 1;
            if (jstep < 0)
                j = ny - j - 1;
            if (kstep < 0)
                k = nz - k - 1;

            // Set kernel args for cell index
            err = clSetKernelArg(k_sweep_cell, 0, sizeof(unsigned int), &i);
            err |= clSetKernelArg(k_sweep_cell, 1, sizeof(unsigned int), &j);
            err |= clSetKernelArg(k_sweep_cell, 2, sizeof(unsigned int), &k);
            check_error(err, "Setting sweep_cell kernel args cell positions");

            // Enqueue the kernel
            err = clEnqueueNDRangeKernel(queue[l % NUM_QUEUES], k_sweep_cell, 2, 0, global, NULL, 0, NULL, NULL);
            check_error(err, "Enqueue sweep_cell kernel");
        }
        // Synchronise command queues at the end of the diagonal plance
        if (NUM_QUEUES > 1)
        {
            for (int q = 0; q < NUM_QUEUES; q++)
            {
                err = clFinish(queue[q]);
                check_error(err, "Finish queues in plane");
            }
        }
    }
}

// Perform a sweep over the grid for all the octants
void ocl_sweep_(void)
{
    cl_int err;

    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    // Get the order of cells to enqueue
    plane *planes = compute_sweep_order();

    for (int o = 0; o < noct; o++)
    {
        enqueue_octant(global_timestep, o, ndiag, planes);
        zero_edge_flux_buffers_();
    }

    for (int i = 0; i < NUM_QUEUES; i++)
    {
        err = clFinish(queue[i]);
        check_error(err, "Finish queue");
    }

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
    if (global_timestep % 2 == 0)
        err = clEnqueueReadBuffer(queue[0], d_flux_out, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, flux_out, 0, NULL, NULL);
    else
        err = clEnqueueReadBuffer(queue[0], d_flux_in, CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*noct*ng, flux_out, 0, NULL, NULL);
    check_error(err, "Reading d_flux_out");
}

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

// Copy the scalar flux value back to the host
void get_scalar_flux_(double *scalar)
{
    cl_int err;
    err = clEnqueueReadBuffer(queue[0], d_scalar_flux, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, scalar, 0, NULL, NULL);
    check_error(err, "Enqueue read scalar_flux buffer");
}

// Set the global timestep variable to the current timestep
void ocl_set_timestep_(const unsigned int *timestep)
{
    global_timestep = (*timestep) - 1;
}
