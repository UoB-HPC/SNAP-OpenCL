
#include "ocl_sweep.h"

// Include the kernel strings
#include "ocl_kernels.h"

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
    if (device_index >= num_devices)
    {
        fprintf(stderr, "Error: Invalid device index %d. Only %d devices available.\n", device_index, num_devices);
        exit(-1);
    }
    device = devices[device_index];

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
    program = clCreateProgramWithSource(context, 1, &ocl_kernels_ocl, NULL, &err);
    check_error(err, "Creating program");

    // Build program
    char *options = "-cl-mad-enable -cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    check_build_error(err, "Building program");

    // Create kernels
    for (int i = 0; i < NUM_QUEUES; i++)
    {
        k_sweep_cell[i] = clCreateKernel(program, "sweep_cell", &err);
        check_error(err, "Creating kernel sweep_cell");
    }

    k_reduce_angular = clCreateKernel(program, "reduce_angular", &err);
    check_error(err, "Creating kernel reduce_angular");

    k_calc_denominator = clCreateKernel(program, "calc_denominator", &err);
    check_error(err, "Creating kernel calc_denominator");

    printf("\nOpenCL environment setup complete\n\n");

}

// Release the global OpenCL handles
void opencl_teardown_(void)
{
    printf("Releasing OpenCL...");

    // Release the event array
    free(events);

    // Release the zero edge array
    free(zero_edge);

    cl_int err;

    // Release all the buffers
    err = clReleaseMemObject(d_source);
    check_error(err, "Releasing d_source buffer");

    for (unsigned int o = 0; o < noct; o++)
    {
        err = clReleaseMemObject(d_flux_in[o]);
        check_error(err, "Releasing d_flux_in buffer");

        err = clReleaseMemObject(d_flux_out[o]);
        check_error(err, "Releasing d_flux_out buffer");
    }

    free(d_flux_in);
    free(d_flux_out);

    err = clReleaseMemObject(d_flux_i);
    check_error(err, "Releasing d_flux_i buffer");

    err = clReleaseMemObject(d_flux_j);
    check_error(err, "Releasing d_flux_j buffer");

    err = clReleaseMemObject(d_flux_k);
    check_error(err, "Releasing d_flux_k buffer");

    err = clReleaseMemObject(d_denom);
    check_error(err, "Releasing d_denom buffer");

    err = clReleaseMemObject(d_dd_j);
    check_error(err, "Releasing d_dd_j buffer");

    err = clReleaseMemObject(d_dd_k);
    check_error(err, "Releasing d_dd_k buffer");

    err = clReleaseMemObject(d_mu);
    check_error(err, "Releasing d_mu buffer");

    err = clReleaseMemObject(d_scat_coeff);
    check_error(err, "Releasing d_scat_coeff buffer");

    err = clReleaseMemObject(d_time_delta);
    check_error(err, "Releasing d_time_delta buffer");

    err = clReleaseMemObject(d_total_cross_section);
    check_error(err, "Releasing d_total_cross_section buffer");

    err = clReleaseMemObject(d_weights);
    check_error(err, "Releasing d_weights buffer");

    err = clReleaseMemObject(d_scalar_flux);
    check_error(err, "Releasing d_scalar_flux buffer");


    for (int i = 0; i < NUM_QUEUES; i++)
    {
        err = clReleaseCommandQueue(queue[i]);
        check_error(err, "Releasing queue");
    }

    // Release kernels
    for (int i = 0; i < NUM_QUEUES; i++)
    {
        err = clReleaseKernel(k_sweep_cell[i]);
        check_error(err, "Releasing k_sweep_cell kernel");
    }

    err = clReleaseKernel(k_reduce_angular);
    check_error(err, "Releasing k_reduce_angular kernel");

    err = clReleaseKernel(k_calc_denominator);
    check_error(err, "Releasing k_calc_denominator kernel");

    // Release program
    err = clReleaseProgram(program);
    check_error(err, "Releasing program");

#ifdef CL_VERSION_1_2
    err = clReleaseDevice(device);
    check_error(err, "Releasing device");
#endif

    // Release context
    err = clReleaseContext(context);
    check_error(err, "Releasing context");

    printf("done\n");
}
