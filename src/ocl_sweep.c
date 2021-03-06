
#include "ocl_sweep.h"



// Forward declare to zero buffer functions
extern void zero_edge_flux_buffers_(void);
extern void zero_centre_flux_in_buffer_(void);

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

// Set all the kernel arguments that are constant over each octant sweep
void set_sweep_cell_args(void)
{
    cl_int err;
    for (int i = 0; i < NUM_QUEUES; i++)
    {
        // Set the kernel arguments
        err = clSetKernelArg(k_sweep_cell[i], 4, sizeof(int), &ichunk);
        err |= clSetKernelArg(k_sweep_cell[i], 5, sizeof(int), &nx);
        err |= clSetKernelArg(k_sweep_cell[i], 6, sizeof(int), &ny);
        err |= clSetKernelArg(k_sweep_cell[i], 7, sizeof(int), &nz);
        err |= clSetKernelArg(k_sweep_cell[i], 8, sizeof(int), &ng);
        err |= clSetKernelArg(k_sweep_cell[i], 9, sizeof(int), &nang);
        err |= clSetKernelArg(k_sweep_cell[i], 10, sizeof(int), &noct);
        err |= clSetKernelArg(k_sweep_cell[i], 11, sizeof(int), &cmom);

        err |= clSetKernelArg(k_sweep_cell[i], 12, sizeof(double), &d_dd_i);
        err |= clSetKernelArg(k_sweep_cell[i], 13, sizeof(cl_mem), &d_dd_j);
        err |= clSetKernelArg(k_sweep_cell[i], 14, sizeof(cl_mem), &d_dd_k);
        err |= clSetKernelArg(k_sweep_cell[i], 15, sizeof(cl_mem), &d_mu);
        err |= clSetKernelArg(k_sweep_cell[i], 16, sizeof(cl_mem), &d_scat_coeff);
        err |= clSetKernelArg(k_sweep_cell[i], 17, sizeof(cl_mem), &d_time_delta);
        err |= clSetKernelArg(k_sweep_cell[i], 18, sizeof(cl_mem), &d_total_cross_section);

        err |= clSetKernelArg(k_sweep_cell[i], 21, sizeof(cl_mem), &d_flux_i);
        err |= clSetKernelArg(k_sweep_cell[i], 22, sizeof(cl_mem), &d_flux_j);
        err |= clSetKernelArg(k_sweep_cell[i], 23, sizeof(cl_mem), &d_flux_k);


        err |= clSetKernelArg(k_sweep_cell[i], 24, sizeof(cl_mem), &d_source);
        err |= clSetKernelArg(k_sweep_cell[i], 25, sizeof(cl_mem), &d_denom);
        err |= clSetKernelArg(k_sweep_cell[i], 27, sizeof(cl_mem), &d_groups_todo);
        check_error(err, "Set sweep_cell kernel args");
    }

}

// Enqueue the kernels to sweep over the grid and compute the angular flux
// Kernel: wavefront of cells
// Work-group: energy group
// Work-item: angle
void enqueue_octant(const unsigned int timestep, const unsigned int oct, const unsigned int ndiag, const plane *planes, const unsigned int num_groups_todo)
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
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;


    cl_int err;

    // Set a local worksize if specified by the environment
    size_t global[2] = {nang * num_groups_todo , 1};
    size_t local_val;
    size_t *local;
    char *local_size = getenv("SNAP_OCL_LOCAL");
    if (local_size != NULL)
    {

        /*local_val = strtol(local_size, NULL, 10);
        local = &local_val;
        printf("Setting local work-group size to %lu\n", local_val);
        // Pad the global size to a multiple of the local size
        if (global[0] % local_val > 0)
        {
            global[0] += local_val - (global[0] % local_val);
            printf("Resetting global size from %d to %lu\n", nang*ng, global[0]);
        }*/
    }
    else
    {
        local = NULL;
    }

    for (int qq = 0; qq < NUM_QUEUES; qq++)
    {
        err = clSetKernelArg(k_sweep_cell[qq], 3, sizeof(unsigned int), &oct);
        err = clSetKernelArg(k_sweep_cell[qq], 0, sizeof(int), &istep);
        err = clSetKernelArg(k_sweep_cell[qq], 1, sizeof(int), &jstep);
        err = clSetKernelArg(k_sweep_cell[qq], 2, sizeof(int), &kstep);
        check_error(err, "Setting octant for sweep_cell kernel");

        err = clSetKernelArg(k_sweep_cell[qq], 28, sizeof(unsigned int), &num_groups_todo);
        check_error(err, "Setting num_groups argument");
        // Swap the angular flux pointers if necessary
        // Even timesteps: Read flux_in and write to flux_out
        // Odd timesteps: Read flux_out and write to flux_in
        if (timestep % 2 == 0)
        {
            err = clSetKernelArg(k_sweep_cell[qq], 19, sizeof(cl_mem), &d_flux_in[oct]);
            err |= clSetKernelArg(k_sweep_cell[qq], 20, sizeof(cl_mem), &d_flux_out[oct]);
        }
        else
        {
            err = clSetKernelArg(k_sweep_cell[qq], 19, sizeof(cl_mem), &d_flux_out[oct]);
            err |= clSetKernelArg(k_sweep_cell[qq], 20, sizeof(cl_mem), &d_flux_in[oct]);
        }
        check_error(err, "Setting flux_in/out args for sweep_cell kernel");
    }

    // Loop over the diagonal wavefronts
    for (unsigned int d = 0; d < ndiag; d++)
    {
        cl_mem d_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(struct cell)*planes[d].num_cells, planes[d].cells, &err);
        check_error(err, "Copying diagonal list");
        err = clSetKernelArg(k_sweep_cell[0], 26, sizeof(cl_mem), &d_list);
        check_error(err, "Setting list argument");
        global[1] = planes[d].num_cells;
        err = clEnqueueNDRangeKernel(queue[0], k_sweep_cell[0], 2, 0, global, local, 0, NULL, NULL);
        check_error(err, "enqueue wavefront");
        err = clReleaseMemObject(d_list);
        check_error(err, "Release mem object");
    }
}

// Perform a sweep over the grid for all the octants
void ocl_sweep_(unsigned int num_groups_todo)
{
    cl_int err;


    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    // Get the order of cells to enqueue
    plane *planes = compute_sweep_order();

    // Set the constant kernel arguemnts
    set_sweep_cell_args();

    for (int o = 0; o < noct; o++)
    {
        enqueue_octant(global_timestep, o, ndiag, planes, num_groups_todo);
        zero_edge_flux_buffers_();
    }

    // The last cell, and the copy zero array are on queue zero,
    // so we only have to wait for this one
    //err = clFinish(queue[0]);
    //check_error(err, "Finish queue 0");

    // Free planes
    for (unsigned int i = 0; i < ndiag; i++)
    {
        free(planes[i].cells);
    }
    free(planes);
}

