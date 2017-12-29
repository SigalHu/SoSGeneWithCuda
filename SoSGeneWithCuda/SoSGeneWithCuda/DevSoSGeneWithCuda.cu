#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "DevSoSGeneWithCuda.cuh"

__global__ void cudaGaussianGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	float omega_n, phi_n, cos_sum;

	for (; y < length; y += y_step){
		cos_sum = 0;
		for (unsigned int xx = x; xx<path_num; xx += x_step){
			omega_n = omega_amp * cosf(delta_alpha * xx) + delta_omega;
			phi_n = 2 * CR_CUDART_PI*dev_uniform[xx];

			cos_value[threadIdx.x] = cosf(omega_n * delta_t*y + phi_n);
			__syncthreads();

			for (unsigned int nn = (path_num <= x_step ? path_num : x_step) >> 1; nn > 0; nn >>= 1){
				if (x < nn){
					cos_value[threadIdx.x] += cos_value[threadIdx.x + nn];
				}
				__syncthreads();
			}

			if (x == 0 && y < length){
				cos_sum += cos_value[threadIdx.x];
			}
		}
		if (x == 0 && y < length){
			dev_vec[y] = sum_amp * cos_sum;
		}
	}
}

__global__ void cudaGaussianGene2(float *dev_vec_i, float *dev_vec_q, unsigned int length, float *dev_uniform,
	unsigned int path_num, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	__shared__ float sin_value[THREAD_NUM_PER_BLOCK];
	float omega_n_i, omega_n_q, phi_n_i, phi_n_q, cos_sum, sin_sum;

	for (; y < length; y += y_step){
		cos_sum = 0;
		sin_sum = 0;
		for (unsigned int xx = x; xx<path_num; xx += x_step){
			omega_n_i = omega_amp * cosf(delta_alpha * xx) + delta_omega;
			omega_n_q = omega_amp * sinf(delta_alpha * xx) + delta_omega;
			phi_n_i = 2 * CR_CUDART_PI*dev_uniform[xx];
			phi_n_q = 2 * CR_CUDART_PI*dev_uniform[xx + path_num];

			cos_value[threadIdx.x] = cosf(omega_n_i * delta_t*y + phi_n_i);
			sin_value[threadIdx.x] = cosf(omega_n_q * delta_t*y + phi_n_q);
			__syncthreads();

			for (unsigned int nn = (path_num <= x_step ? path_num : x_step) >> 1; nn > 0; nn >>= 1){
				if (x < nn){
					cos_value[threadIdx.x] += cos_value[threadIdx.x + nn];
					sin_value[threadIdx.x] += sin_value[threadIdx.x + nn];
				}
				__syncthreads();
			}

			if (x == 0 && y < length){
				cos_sum += cos_value[threadIdx.x];
				sin_sum += sin_value[threadIdx.x];
			}
		}
		if (x == 0 && y < length){
			dev_vec_i[y] = sum_amp * cos_sum;
			dev_vec_q[y] = sum_amp * sin_sum;
		}
	}
}

__global__ void cudaLognormalGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	float omega_n, phi_n, cos_sum;

	for (; y < length; y += y_step){
		cos_sum = 0;
		for (unsigned int xx = x; xx<path_num; xx += x_step){
			omega_n = omega_amp * cosf(delta_alpha * xx) + delta_omega;
			phi_n = 2 * CR_CUDART_PI*dev_uniform[xx];

			cos_value[threadIdx.x] = cosf(omega_n * delta_t*y + phi_n);
			__syncthreads();

			for (unsigned int nn = (path_num <= x_step ? path_num : x_step) >> 1; nn > 0; nn >>= 1){
				if (x < nn){
					cos_value[threadIdx.x] += cos_value[threadIdx.x + nn];
				}
				__syncthreads();
			}

			if (x == 0 && y < length){
				cos_sum += cos_value[threadIdx.x];
			}
		}
		if (x == 0 && y < length){
			dev_vec[y] = __expf(sum_amp * cos_sum);
		}
	}
}

__global__ void cudaNakagamiGene(float *dev_vec, unsigned int length, float *dev_uniform, 
	unsigned int path_num, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp, bool is_end){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	float omega_n, phi_n, cos_sum;

	for (; y < length; y += y_step){
		cos_sum = 0;
		for (unsigned int xx = x; xx<path_num; xx += x_step){
			omega_n = omega_amp * __cosf(delta_alpha * xx) + delta_omega;
			phi_n = 2 * CR_CUDART_PI*dev_uniform[xx];

			cos_value[threadIdx.x] = __cosf(omega_n * delta_t*y + phi_n);
			__syncthreads();

			for (unsigned int mm = (path_num <= x_step ? path_num : x_step) >> 1; mm > 0; mm >>= 1){
				if (x < mm){
					cos_value[threadIdx.x] += cos_value[threadIdx.x + mm];
				}
				__syncthreads();
			}

			if (x == 0 && y < length){
				cos_sum += cos_value[threadIdx.x];
			}
		}
		if (x == 0 && y < length){
			if (is_end){
				dev_vec[y] = sqrtf(dev_vec[y] + sum_amp * cos_sum * cos_sum);
			}
			else {
				dev_vec[y] += sum_amp * cos_sum * cos_sum;
			}
		}
		__threadfence_system();
	}
}

__global__ void cudaNakagamiGene2(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp, unsigned int gaussian_n){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	float omega_n, phi_n, cos_sum, result;

	for (; y < length; y += y_step){
		result = 0;
		for (unsigned int nn = 0; nn < gaussian_n; nn++){
			cos_sum = 0;
			for (unsigned int xx = x; xx<path_num; xx += x_step){
				omega_n = omega_amp * __cosf(delta_alpha * xx) + delta_omega;
				phi_n = 2 * CR_CUDART_PI*dev_uniform[xx + nn*path_num];

				cos_value[threadIdx.x] = __cosf(omega_n * delta_t*y + phi_n);
				__syncthreads();

				for (unsigned int mm = (path_num <= x_step ? path_num : x_step) >> 1; mm > 0; mm >>= 1){
					if (x < mm){
						cos_value[threadIdx.x] += cos_value[threadIdx.x + mm];
					}
					__syncthreads();
				}
				if (x == 0 && y < length){
					cos_sum += cos_value[threadIdx.x];
				}
			}
			if (x == 0 && y < length){
				result += sum_amp * cos_sum * cos_sum;
			}
		}
		if (x == 0 && y < length){
			dev_vec[y] = sqrtf(result);
		}
	}
}

__global__ void cudaLogNakGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num, float mean,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float log_sum_amp, float nak_sum_amp, unsigned int gaussian_n){
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int x = tidx % path_num;
	unsigned int x_step = blockDim.x;

	unsigned int y = tidy;
	unsigned int y_step = gridDim.y;
	if (path_num < x_step){
		y = (y * x_step + tidx) / path_num;
		y_step *= x_step / path_num;
	}

	__shared__ float cos_value[THREAD_NUM_PER_BLOCK];
	float omega_n, phi_n, cos_sum, result;

	for (; y < length; y += y_step){
		result = 0;
		for (unsigned int nn = 0; nn < gaussian_n; nn++){
			cos_sum = 0;
			for (unsigned int xx = x; xx<path_num; xx += x_step){
				omega_n = omega_amp * __cosf(delta_alpha * xx) + delta_omega;
				phi_n = 2 * CR_CUDART_PI*dev_uniform[xx + nn*path_num];

				cos_value[threadIdx.x] = __cosf(omega_n * delta_t*y + phi_n);
				__syncthreads();

				for (unsigned int mm = (path_num <= x_step ? path_num : x_step) >> 1; mm > 0; mm >>= 1){
					if (x < mm){
						cos_value[threadIdx.x] += cos_value[threadIdx.x + mm];
					}
					__syncthreads();
				}
				if (x == 0 && y < length){
					cos_sum += cos_value[threadIdx.x];
				}
			}
			if (x == 0 && y < length){
				if (nn < gaussian_n - 1){
					result += nak_sum_amp * cos_sum * cos_sum;
				}
				else{
					//dev_vec[y] = __expf(log_sum_amp*cos_sum + mean);
					//dev_vec[y] = sqrtf(result);
					dev_vec[y] = sqrtf(result * __expf(log_sum_amp*cos_sum + mean));
				}
				
			}
		}
	}
}