
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sweep_cell(
	// Current cell index
	const unsigned int i,
	const unsigned int j,
	const unsigned int k,

	// Angular flux
	__global double *flux_in,
	__global double *flux_out,

	// Source
	__global double *source,
	__global double *denom,

	// Flux halos
	__global double *flux_halo_y,
	__global double *flux_halo_z
	)
{
	int idx = get_global_id(0);
	flux_out[idx] = flux_in[idx];
	return;
}
