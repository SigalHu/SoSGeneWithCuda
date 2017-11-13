#include <ctime>
#include <string>
#include "CudaRandUtils.cuh"

namespace CudaRandUtils{
	std::string getStatusStr(curandStatus_t status){
		switch (status){
		case CURAND_STATUS_SUCCESS:
			return "No errors.";
		case CURAND_STATUS_VERSION_MISMATCH:
			return "Header file and linked library version do not match.";
		case CURAND_STATUS_NOT_INITIALIZED:
			return "Generator not initialized.";
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "Memory allocation failed.";
		case CURAND_STATUS_TYPE_ERROR:
			return "Generator is wrong type.";
		case CURAND_STATUS_OUT_OF_RANGE:
			return "Argument out of range.";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "Length requested is not a multple of dimension.";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "GPU does not have double precision required by MRG32k3a.";
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "Kernel launch failure.";
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "Preexisting failure on library entry.";
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "Initialization of CUDA failed.";
		case CURAND_STATUS_ARCH_MISMATCH:
			return "Architecture mismatch, GPU does not support requested feature.";
		case CURAND_STATUS_INTERNAL_ERROR:
			return "Internal library error.";
		default:
			return "unrecognized error code.";
		}
	}

	bool createGenerator(curandGenerator_t &generator, curandRngType_t rng_type){
		curandStatus_t status = curandCreateGenerator(&generator, rng_type);
		if (CURAND_STATUS_SUCCESS != status){
			std::cerr << "#######################################" << std::endl;
			std::cerr << "curandCreateGenerator launch failed!" << std::endl;
			std::cerr << getStatusStr(status) << std::endl;
			std::cerr << "#######################################" << std::endl;
			return false;
		}

		status = curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
		if (CURAND_STATUS_SUCCESS != status){
			std::cerr << "#######################################" << std::endl;
			std::cerr << "curandSetPseudoRandomGeneratorSeed launch failed!" << std::endl;
			std::cerr << getStatusStr(status) << std::endl;
			std::cerr << "#######################################" << std::endl;
			return false;
		}
		return true;
	}

	bool generateNormal(thrust::device_vector<float> &dv, float mean, float stddev){
		curandGenerator_t generator;
		if (!createGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT))
			return false;

		float *p = thrust::raw_pointer_cast(dv.data());
		unsigned int len = dv.size();
		curandStatus_t status = curandGenerateNormal(generator, p, len, mean, stddev);
		if (CURAND_STATUS_SUCCESS != status){
			std::cerr << "#######################################" << std::endl;
			std::cerr << "curandGenerateNormal launch failed!" << std::endl;
			std::cerr << getStatusStr(status) << std::endl;
			std::cerr << "#######################################" << std::endl;
			return false;
		}
		return true;
	}

	bool generateUniform(thrust::device_vector<float> &dv){
		curandGenerator_t generator;
		if (!createGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT))
			return false;

		float *p = thrust::raw_pointer_cast(dv.data());
		unsigned int len = dv.size();
		curandStatus_t status = curandGenerateUniform(generator, p, len);
		if (CURAND_STATUS_SUCCESS != status){
			std::cerr << "#######################################" << std::endl;
			std::cerr << "curandGenerateUniform launch failed!" << std::endl;
			std::cerr << getStatusStr(status) << std::endl;
			std::cerr << "#######################################" << std::endl;
			return false;
		}
		return true;
	}
}