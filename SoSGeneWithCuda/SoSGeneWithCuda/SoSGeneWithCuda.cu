#include "device_launch_parameters.h"
#include "DevSoSGeneWithCuda.cuh"
#include "GridDimUtils.cuh"
#include "CudaRandUtils.cuh"
#include "SoSGeneWithCuda.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <map>
#include <iostream>

static const int THREAD_NUM_PER_BLOCK = 256;

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(thrust::device_vector<int> &c, thrust::device_vector<int> &a, thrust::device_vector<int> &b, unsigned int size)
{
	int *dev_a = thrust::raw_pointer_cast(a.data());
	int *dev_b = thrust::raw_pointer_cast(b.data());
	int *dev_c = thrust::raw_pointer_cast(c.data());
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

bool noiseGene(const unsigned int &deviceId, thrust::device_vector<float> &dvNoiseI, thrust::device_vector<float> &dvNoiseQ, const float &fs,
	const float &avgPower, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega){

	if (dvNoiseI.empty() || dvNoiseQ.empty())
		return false;
	if (dvNoiseI.size() != dvNoiseQ.size())
		return false;
	unsigned int tmp = 1;
	while (tmp < pathNum)
		tmp <<= 1;
	if (pathNum != tmp)
		return false;

	float omegaAmp = 2 * M_PI*maxFd;
	float deltaAlpha = (2 * M_PI - 2 * M_PI / (pathNum + 1)) / (pathNum - 1);
	float deltaT = 1 / fs;
	float sumAmp = sqrtf(avgPower / pathNum);

	thrust::device_vector<float> dvUniform(2 * pathNum);
	CudaRandUtils::generateUniform(dvUniform);

	size_t len = dvNoiseI.size();
	dim3 blockNum, threadNum;
	threadNum.x = THREAD_NUM_PER_BLOCK;
	if (pathNum <= THREAD_NUM_PER_BLOCK){
		blockNum.y = len;
	}
	else{
		blockNum.x = pathNum / THREAD_NUM_PER_BLOCK;
		blockNum.y = len / blockNum.x + (len % blockNum.x) != 0;
	}
	const dim3 gridDim = GridDimUtils::getGridDim(deviceId);
	if (blockNum.y > gridDim.y){
		blockNum.y = gridDim.y;
	}

	float *pNoiseI = thrust::raw_pointer_cast(dvNoiseI.data());
	float *pNoiseQ = thrust::raw_pointer_cast(dvNoiseQ.data());
	float *pUniform = thrust::raw_pointer_cast(dvUniform.data());
	cudaNoiseGeneWithSoS << <blockNum, threadNum, 2 * pathNum*sizeof(float) >> >(
		pNoiseI, pNoiseQ, len, pUniform, pathNum, time(NULL), omegaAmp, deltaAlpha, deltaOmega, deltaT, sumAmp);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess){
		std::cerr << "#######################################" << std::endl;
		std::cerr << "cudaNoiseGeneWithSoS launch failed!" << std::endl;
		std::cerr << cudaGetErrorString(error) << std::endl;
		std::cerr << "#######################################" << std::endl;
		return false;
	}

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
		std::cerr << "#######################################" << std::endl;
		std::cerr << "cudaDeviceSynchronize launch failed!" << std::endl;
		std::cerr << cudaGetErrorString(error) << std::endl;
		std::cerr << "#######################################" << std::endl;
		return false;
	}
	return true;
}