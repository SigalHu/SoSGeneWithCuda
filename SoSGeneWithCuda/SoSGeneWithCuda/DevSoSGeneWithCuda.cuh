#pragma once
#include "cuda_runtime.h"

__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void cudaNoiseGeneWithSoS(float *dev_cos_value, float *dev_sin_value, unsigned int length, float *dev_uniform, unsigned int path_num,
	unsigned long long uniform_seed, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp);