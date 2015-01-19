
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define flux_out(a,i,j,k,o,g) flux_out[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)]

__kernel void sweep_cell(
	// Current cell index
	const unsigned int i,
	const unsigned int j,
	const unsigned int k,

	// Problem sizes
	const unsigned int ichunk,
	const unsigned int nx,
	const unsigned int ny,
	const unsigned int nz,
	const unsigned int ng,
	const unsigned int nang,
	const unsigned int noct,

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
	printf("%d %d %d\n", i,j,k);
	int a_idx = get_global_id(0);
	int g_idx = get_global_id(1);
	int oct = 0;
	printf("%d - %d\n", a_idx, g_idx);
	flux_out(a_idx,i,j,k,oct,g_idx) = 12345.0;
	return;
}
