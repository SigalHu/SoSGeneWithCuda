#pragma once
#include "cuda_runtime.h"
#include <map>

namespace CudaCommonUtils{
	const dim3 getGridDim(const unsigned int &dev_id);
	void printDeviceProperties();
}
