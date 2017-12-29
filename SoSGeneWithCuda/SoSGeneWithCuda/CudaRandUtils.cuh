#pragma once
#include "curand.h"

namespace CudaRandUtils{
	bool generateNormal(float *dev_array, size_t length, float mean, float stddev);
	bool generateUniform(float *dev_array, size_t length);
}