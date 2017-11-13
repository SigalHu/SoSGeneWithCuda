#include "thrust\host_vector.h"
#include "SoSGeneWithCuda.h"
#include <iostream>

int main()
{
	unsigned int deviceId = 0;
	cudaError_t cudaStatus = cudaSetDevice(deviceId);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
		std::cin.get();
		return 0;
	}

	{
		const float fs = 1000;
		const float time_spend = 1000;
		const unsigned int len = (unsigned int)(fs*time_spend);
		thrust::device_vector<float> devNoiseI(len);
		thrust::device_vector<float> devNoiseQ(len);

		if (!noiseGene(deviceId, devNoiseI, devNoiseQ, fs)){
			std::cout << "noiseGene failed!" << std::endl;
			std::cin.get();
			return 1;
		}
		std::cout << "noiseGene succeed!" << std::endl;

		thrust::host_vector<float> hostNoiseI(devNoiseI);
		thrust::host_vector<float> hostNoiseQ(devNoiseQ);

		std::cout << hostNoiseI.front() << "->" << hostNoiseI.back() << std::endl;
		for (int ii = 0; ii < hostNoiseI.size(); ++ii){
			if (ii != hostNoiseI[ii]){
				std::cout << "is false!" << std::endl;
				break;
			}
		}
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!" << std::endl;
		std::cin.get();
		return 1;
	}
	std::cin.get();
	return 0;
}
