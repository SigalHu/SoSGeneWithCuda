#include <iostream>
#include "GridDimUtils.cuh"

namespace GridDimUtils{
	std::map<unsigned int, const dim3> getGridDimList();

	static std::map<unsigned int, const dim3> gridDimMap = GridDimUtils::getGridDimList();

	std::map<unsigned int, const dim3> getGridDimList(){
		int count = 0;
		cudaDeviceProp prop;
		std::map<unsigned int, const dim3> map;

		cudaError_t error = cudaGetDeviceCount(&count);
		if (error != cudaSuccess){
			std::cerr << "#######################################" << std::endl;
			std::cerr << "cudaGetDeviceCount launch failed!" << std::endl;
			std::cerr << cudaGetErrorString(error) << std::endl;
			std::cerr << "#######################################" << std::endl;
			return map;
		}

		for (unsigned int ii = 0; ii < count; ++ii){
			if (cudaGetDeviceProperties(&prop, ii) != cudaSuccess){
				std::cerr << "#######################################" << std::endl;
				std::cerr << "cudaGetDeviceProperties(ii = "<< ii << ")launch failed!" << std::endl;
				std::cerr << cudaGetErrorString(error) << std::endl;
				std::cerr << "#######################################" << std::endl;
			}
			else
				map.emplace(ii, dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]));
		}
		return map;
	}

	const dim3 getGridDim(const unsigned int &deviceId){
		std::map<unsigned int, const dim3>::iterator iter = gridDimMap.find(deviceId);
		if (iter == gridDimMap.end()){
			dim3 defaultDim(65535, 65535, 65535);
			std::cerr << "#######################################" << std::endl;
			std::cerr << "deviceId = " << deviceId << " is not found!" << std::endl;
			std::cerr << "using default value: dim3={" << defaultDim.x << ", " << defaultDim.y << ", " << defaultDim.z << "}" << std::endl;
			std::cerr << "#######################################" << std::endl;
			return defaultDim;
		}
		return iter->second;
	}
}