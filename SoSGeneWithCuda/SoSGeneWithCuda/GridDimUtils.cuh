#pragma once
#include "cuda_runtime.h"
#include <map>

namespace GridDimUtils{
	const dim3 getGridDim(const unsigned int &deviceId);
}
