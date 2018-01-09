#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "DevSoSGeneWithCuda.cuh"
#include "CudaCommonUtils.cuh"
#include "CudaRandUtils.cuh"
#include "SoSGeneWithCuda.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <map>
#include <iostream>

bool gaussianGene(float *&arr, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &mean, const float &variance, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = sqrtf(2 * variance / path_num);

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, path_num))
			throw false;
		
		// ����GPU����
		cudaGaussianGene << <block_num, thread_num >> >(
			dev_arr, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaGaussianGene����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr, (void *)dev_arr, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr)
		cudaFree(dev_arr);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool gaussianGeneIQ(float *&arr_i, float *&arr_q, size_t &length, const float &fs, const float &fd_max,
	const float &time_spend, const unsigned int &path_num, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr_i = NULL;
	float *dev_arr_q = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = sqrtf(2.0 / path_num);

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr_i = new float[length];
		arr_q = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr_i, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_arr_q, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, 2 * path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, 2 * path_num))
			throw false;

		// ����GPU����
		cudaGaussianGeneIQ << <block_num, thread_num >> >(
			dev_arr_i, dev_arr_q, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaGaussianGene����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_i, (void *)dev_arr_i, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_q, (void *)dev_arr_q, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr_i)
		cudaFree(dev_arr_i);
	if (dev_arr_q)
		cudaFree(dev_arr_q);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool lognormalGene(float *&arr, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &mean, const float &variance, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = sqrtf(2 * variance / path_num);

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, path_num))
			throw false;

		// ����GPU����
		cudaLognormalGene << <block_num, thread_num >> >(
			dev_arr, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaLognormalGene����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr, (void *)dev_arr, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr)
		cudaFree(dev_arr);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool nakagamiGene(float *&arr, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &nak_m, const float &nak_omega, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float mean = 0;
		float variance = nak_omega / (2 * nak_m);
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = 2 * variance / path_num;
		unsigned int gaussian_n = 2 * nak_m + 0.5;

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemset(dev_arr, 0, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "��ʼ��GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		for (unsigned int nn = gaussian_n; nn > 0; nn--){
			// ���������
			if (!CudaRandUtils::generateUniform(dev_uniform, path_num))
				throw false;

			// ����GPU����
			cudaNakagamiGene << <block_num, thread_num >> >(
				dev_arr, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp, nn==1);
			_sleep(1000);
			status = cudaGetLastError();
			if (status != cudaSuccess){
				std::cerr << "cudaLognormalGene����ʧ�ܣ�" << std::endl;
				std::cerr << cudaGetErrorString(status) << std::endl;
				throw false;
			}

			status = cudaDeviceSynchronize();
			if (status != cudaSuccess){
				std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
				std::cerr << cudaGetErrorString(status) << std::endl;
				throw false;
			}
		}

		status = cudaMemcpy((void *)arr, (void *)dev_arr, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr)
		cudaFree(dev_arr);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool nakagamiGene2(float *&arr, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &nak_m, const float &nak_omega, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float mean = 0;
		float variance = nak_omega / (2 * nak_m);
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = 2 * variance / path_num;
		unsigned int gaussian_n = 2 * nak_m + 0.5;

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, gaussian_n*path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, gaussian_n*path_num))
			throw false;

		// ����GPU����
		cudaNakagamiGene2 << <block_num, thread_num>> >(
			dev_arr, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp, gaussian_n);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaLognormalGene2����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr, (void *)dev_arr, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr)
		cudaFree(dev_arr);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool nakagamiGeneIQ(float *&arr_i, float *&arr_q, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &nak_m, const float &nak_omega, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr_i = NULL;
	float *dev_arr_q = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float mean = 0;
		float variance = nak_omega / (2 * nak_m);
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;
		float sum_amp = 2 * variance / path_num;

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr_i = new float[length];
		arr_q = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr_i, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_arr_q, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, 2 * nak_m*path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, 2 * nak_m*path_num))
			throw false;

		// ����GPU����
		cudaNakagamiGeneIQ << <block_num, thread_num >> >(
			dev_arr_i, dev_arr_q, length, dev_uniform, path_num, omega_amp, delta_alpha, delta_omega, delta_t, sum_amp, nak_m);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaLognormalGeneIQ����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_i, (void *)dev_arr_i, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_q, (void *)dev_arr_q, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr_i)
		cudaFree(dev_arr_i);
	if (dev_arr_q)
		cudaFree(dev_arr_q);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool lognakGene(float *&arr, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &nak_m, const float &shadow_db, const float &power_avg, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float log_mean = logf(power_avg);
		float log_variance = shadow_db*shadow_db / 8.686/ 8.686;
		float log_sum_amp = sqrtf(2 * log_variance / path_num);

		float nak_omega = 1;
		float nak_mean = 0;
		float nak_variance = nak_omega / (2 * nak_m);
		float nak_sum_amp = 2 * nak_variance / path_num;
		
		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;

		unsigned int gaussian_n = 2 * nak_m + 1.5;

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, gaussian_n*path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, gaussian_n*path_num))
			throw false;

		// ����GPU����
		cudaLogNakGene << <block_num, thread_num >> >(
			dev_arr, length, dev_uniform, path_num, log_mean, omega_amp, delta_alpha, delta_omega, delta_t, log_sum_amp, nak_sum_amp, gaussian_n);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaLogNakGene����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr, (void *)dev_arr, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr)
		cudaFree(dev_arr);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}

bool lognakGeneIQ(float *&arr_i, float *&arr_q, size_t &length, const float &fs, const float &fd_max, const float &time_spend,
	const unsigned int &path_num, const float &nak_m, const float &shadow_db, const float &power_avg, const float &delta_omega){
	bool is_succeed = true;
	cudaError_t status;
	float *dev_arr_i = NULL;
	float *dev_arr_q = NULL;
	float *dev_uniform = NULL;

	try{
		// �������
		unsigned int tmp = 1;
		while (tmp < path_num)
			tmp <<= 1;
		if (path_num != tmp){
			std::cerr << "path_numȡֵ����Ϊ2�����η���" << std::endl;
			throw false;
		}

		// ����Ԥ����
		float log_mean = logf(power_avg);
		float log_variance = shadow_db*shadow_db / 8.686 / 8.686;
		float log_sum_amp = sqrtf(2 * log_variance / path_num);

		float nak_omega = 1;
		float nak_mean = 0;
		float nak_variance = nak_omega / (2 * nak_m);
		float nak_sum_amp = 2 * nak_variance / path_num;

		float omega_amp = 2 * M_PI*fd_max;
		float delta_alpha = (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1);
		float delta_t = 1 / fs;

		// ѡ��GPU
		const unsigned int dev_id = 0;
		status = cudaSetDevice(dev_id);
		if (status != cudaSuccess) {
			std::cerr << "ѡ��GPU�豸ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		length = (size_t)(fs*time_spend);
		arr_i = new float[length];
		arr_q = new float[length];
		// hu ����ռ�
		status = cudaMalloc((void **)&dev_arr_i, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_arr_q, length*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMalloc((void **)&dev_uniform, (2 * nak_m + 1)*path_num*sizeof(float));
		if (status != cudaSuccess) {
			std::cerr << "����GPU�ڴ�ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		// ȷ���̲߳���
		dim3 block_num, thread_num;
		thread_num.x = THREAD_NUM_PER_BLOCK;
		if (path_num < THREAD_NUM_PER_BLOCK){
			block_num.y = (length - 1) / (THREAD_NUM_PER_BLOCK / path_num) + 1;
		}
		else{
			block_num.y = length;
		}
		const dim3 grid_dim = CudaCommonUtils::getGridDim(dev_id);
		if (block_num.y > grid_dim.y){
			block_num.y = grid_dim.y;
		}

		// ���������
		if (!CudaRandUtils::generateUniform(dev_uniform, (2 * nak_m + 1)*path_num))
			throw false;

		// ����GPU����
		cudaLogNakGeneIQ << <block_num, thread_num >> >(
			dev_arr_i, dev_arr_q, length, dev_uniform, path_num, log_mean, omega_amp, delta_alpha, delta_omega, delta_t, log_sum_amp, nak_sum_amp, nak_m);
		status = cudaGetLastError();
		if (status != cudaSuccess){
			std::cerr << "cudaLogNakGeneIQ����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess){
			std::cerr << "cudaDeviceSynchronize����ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_i, (void *)dev_arr_i, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}

		status = cudaMemcpy((void *)arr_q, (void *)dev_arr_q, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			std::cerr << "GPU->CPU���ݴ���ʧ�ܣ�" << std::endl;
			std::cerr << cudaGetErrorString(status) << std::endl;
			throw false;
		}
	}
	catch (bool &ex){
		is_succeed = ex;
	}

	if (dev_arr_i)
		cudaFree(dev_arr_i);
	if (dev_arr_q)
		cudaFree(dev_arr_q);
	if (dev_uniform)
		cudaFree(dev_uniform);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		std::cerr << "GPU��λʧ�ܣ�" << std::endl;
		std::cerr << cudaGetErrorString(status) << std::endl;
		return false;
	}
	return is_succeed;
}