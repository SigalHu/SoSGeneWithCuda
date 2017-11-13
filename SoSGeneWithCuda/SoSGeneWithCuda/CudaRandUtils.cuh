#pragma once
#include "curand.h"
#include "thrust\device_vector.h"

namespace CudaRandUtils{
	bool generateNormal(thrust::device_vector<float> &dv, float mean, float stddev);
	bool generateUniform(thrust::device_vector<float> &dv);
}