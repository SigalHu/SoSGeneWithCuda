#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "DevSoSGeneWithCuda.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void cudaNoiseGeneWithSoS(float *dev_cos_value, float *dev_sin_value, unsigned int length, float *dev_uniform, unsigned int path_num,
	unsigned long long uniform_seed, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp){
	extern __shared__ float _sha[];
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int y = (tidy * blockDim.x * gridDim.x + tidx) / path_num;

	float omega_n_I = omega_amp * cosf(delta_alpha * x) + delta_omega;
	float omega_n_Q = omega_amp * sinf(delta_alpha * x) + delta_omega;

	//curandState_t rand_status;
	//curand_init(uniform_seed, x, 0, &rand_status);
	//float phi_n_I = curand_uniform(&rand_status);
	//float phi_n_Q = curand_uniform(&rand_status);
	float phi_n_I = dev_uniform[x];
	float phi_n_Q = dev_uniform[x + path_num];

	float *cos_value = _sha, *sin_value = _sha + path_num;
	unsigned int y_step = gridDim.y * blockDim.x * gridDim.x / path_num;
	for (; y < length; y += y_step){
		cos_value[x] = cosf(omega_n_I * delta_t*y + 2 * CR_CUDART_PI*phi_n_I);
		sin_value[x] = sinf(omega_n_Q * delta_t*y + 2 * CR_CUDART_PI*phi_n_Q);
		__syncthreads();

		for (path_num >>= 1; path_num > 0; path_num >>= 1){
			if (x < path_num){
				cos_value[x] += cos_value[x + path_num];
				sin_value[x] += sin_value[x + path_num];
			}
			__syncthreads();
		}

		if (x == 0){
			dev_cos_value[y] = sum_amp * cos_value[x];
			dev_sin_value[y] = sum_amp * sin_value[x];
		}
		__syncthreads();
	}
}