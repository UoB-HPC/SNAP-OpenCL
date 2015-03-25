
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <omp.h>

#define NUM_QUEUES 4

#include "ocl_problem.h"

// Global OpenCL handles (context, queue, etc.)
cl_device_id device;
cl_context context;
cl_command_queue queue[NUM_QUEUES];
cl_program program;

// OpenCL kernels
cl_kernel k_sweep_cell[NUM_QUEUES];
cl_kernel k_reduce_angular;

// OpenCL buffers
cl_mem d_source;
cl_mem d_flux_in;
cl_mem d_flux_out;
cl_mem d_flux_i;
cl_mem d_flux_j;
cl_mem d_flux_k;
cl_mem d_denom;
double d_dd_i;
cl_mem d_dd_j;
cl_mem d_dd_k;
cl_mem d_mu;
cl_mem d_scat_coeff;
cl_mem d_time_delta;
cl_mem d_total_cross_section;
cl_mem d_weights;
cl_mem d_scalar_flux;


// List of OpenCL events, one for each cell
// This is used to encourage spacial parallelism by
// enqueuing kernels in multiple queues which only
// depend on all their downwind neighbours.
// This list stores those events, and is reset every octant.
cl_event *events;

// Create an empty buffer to zero out the edge flux arrays
// Each direction can share it as we make sure that it is
// big enough for each of them
double *zero_edge;

// Global variable for the timestep
unsigned int global_timestep;

// Check OpenCL errors and exit if no success
#define check_error(e,m) __check_error(e,m,__LINE__,__FILE__)
static void __check_error(cl_int err, char *msg, int line, char *file)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error %d: %s on line %d in %s\n", err, msg, line, file);
        exit(err);
    }
};

// Check for OpenCL build errors and display build messages
static void check_build_error(cl_int err, char *msg)
{
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        char *build_log = (char*)malloc(sizeof(char)*4096);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*4096, build_log, NULL);
        fprintf(stderr, "Error: %d\n", err);
        fprintf(stderr, "Build log:\n%s\n", build_log);
        free(build_log);
    }
    check_error(err, msg);
};

