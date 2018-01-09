#pragma once
#include "cuda_runtime.h"

static const int THREAD_NUM_PER_BLOCK = 256;

__global__ void cudaGaussianGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp);

__global__ void cudaGaussianGeneIQ(float *dev_vec_i, float *dev_vec_q, unsigned int length, float *dev_uniform,
	unsigned int path_num, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp);

__global__ void cudaLognormalGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp);

__global__ void cudaNakagamiGene(float *dev_vec, unsigned int length, float *dev_uniform,
	unsigned int path_num, float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp, bool is_end);

__global__ void cudaNakagamiGene2(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp, unsigned int gaussian_n);

__global__ void cudaNakagamiGeneIQ(float *dev_vec_i, float *dev_vec_q, unsigned int length, float *dev_uniform, unsigned int path_num,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float sum_amp, unsigned int nak_m);

__global__ void cudaLogNakGene(float *dev_vec, unsigned int length, float *dev_uniform, unsigned int path_num, float mean,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float log_sum_amp, float nak_sum_amp, unsigned int gaussian_n);

__global__ void cudaLogNakGeneIQ(float *dev_vec_i, float *dev_vec_q, unsigned int length, float *dev_uniform, unsigned int path_num, float mean,
	float omega_amp, float delta_alpha, float delta_omega, float delta_t, float log_sum_amp, float nak_sum_amp, unsigned int nak_m);