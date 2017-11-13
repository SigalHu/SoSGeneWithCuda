#pragma once
#include "cuda_runtime.h"
#include "thrust\device_vector.h"

cudaError_t addWithCuda(thrust::device_vector<int> &c, thrust::device_vector<int> &a, thrust::device_vector<int> &b, unsigned int size);

bool noiseGene(const unsigned int &deviceId, thrust::device_vector<float> &dvNoiseI, thrust::device_vector<float> &dvNoiseQ, const float &fs,
	const float &avgPower = 1, const unsigned int &pathNum = 32, const float &maxFd = 50, const float &deltaOmega = 0);