#include <iostream>
#include "CudaCommonUtils.cuh"

namespace CudaCommonUtils{
	std::map<unsigned int, const dim3> getGridDimList();

	static std::map<unsigned int, const dim3> gridDimMap = CudaCommonUtils::getGridDimList();

	std::map<unsigned int, const dim3> getGridDimList(){
		int count = 0;
		cudaDeviceProp prop;
		std::map<unsigned int, const dim3> map;

		cudaError_t error = cudaGetDeviceCount(&count);
		if (error != cudaSuccess){
			std::cerr << "cudaGetDeviceCount调用失败！" << std::endl;
			std::cerr << cudaGetErrorString(error) << std::endl;
			return map;
		}

		for (unsigned int ii = 0; ii < count; ++ii){
			error = cudaGetDeviceProperties(&prop, ii);
			if (error != cudaSuccess){
				std::cerr << "cudaGetDeviceProperties(ii = "<< ii << ")调用失败！" << std::endl;
				std::cerr << cudaGetErrorString(error) << std::endl;
			}
			else
				map.emplace(ii, dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]));
		}
		return map;
	}

	const dim3 getGridDim(const unsigned int &dev_id){
		std::map<unsigned int, const dim3>::iterator iter = gridDimMap.find(dev_id);
		if (iter == gridDimMap.end()){
			dim3 defaultDim(65535, 65535, 65535);
			std::cerr << "dev_id = " << dev_id << "GPU设备不存在！" << std::endl;
			std::cerr << "使用默认值：dim3={" << defaultDim.x << ", " << defaultDim.y << ", " << defaultDim.z << "}" << std::endl;
			return defaultDim;
		}
		return iter->second;
	}

	void printDeviceProperties(){
		cudaDeviceProp prop;
		cudaError_t error = cudaGetDeviceProperties(&prop, 0);
		if (cudaSuccess != error){
			std::cerr << "cudaGetDeviceProperties调用失败！" << std::endl;
			std::cerr << cudaGetErrorString(error) << std::endl;
		}

		printf("###############################################\n");
		printf("Device Name : %s.\n", prop.name);
		printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
		printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
		printf("regsPerBlock : %d.\n", prop.regsPerBlock);
		printf("warpSize : %d.\n", prop.warpSize);
		printf("memPitch : %d.\n", prop.memPitch);
		printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
		printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("totalConstMem : %d.\n", prop.totalConstMem);
		printf("major.minor : %d.%d.\n", prop.major, prop.minor);
		printf("clockRate : %d.\n", prop.clockRate);
		printf("textureAlignment : %d.\n", prop.textureAlignment);
		printf("deviceOverlap : %d.\n", prop.deviceOverlap);
		printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
		printf("###############################################\n");
	}
}