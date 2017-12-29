#pragma once

bool gaussianGene(float *&arr, size_t &length, const float &fs = 1000, const float &fd_max = 50, const float &time_spend = 1,
	const unsigned int &path_num = 32, const float &mean = 0, const float &variance = 1, const float &delta_omega = 0);

bool gaussianGene2(float *&arr_i, float *&arr_q, size_t &length, const float &fs = 1000, const float &fd_max = 50, 
	const float &time_spend = 1, const unsigned int &path_num = 32, const float &delta_omega = 0);

bool lognormalGene(float *&arr, size_t &length, const float &fs = 1000, const float &fd_max = 50, const float &time_spend = 1,
	const unsigned int &path_num = 32, const float &mean = 0, const float &variance = 1, const float &delta_omega = 0);

bool nakagamiGene(float *&arr, size_t &length, const float &fs = 1000, const float &fd_max = 50, const float &time_spend = 1,
	const unsigned int &path_num = 32, const float &nak_m = 1, const float &nak_omega = 1, const float &delta_omega = 0);

bool nakagamiGene2(float *&arr, size_t &length, const float &fs = 1000, const float &fd_max = 50, const float &time_spend = 1,
	const unsigned int &path_num = 32, const float &nak_m = 1, const float &nak_omega = 1, const float &delta_omega = 0);

bool lognakGene(float *&arr, size_t &length, const float &fs = 1000, const float &fd_max = 50, const float &time_spend = 1, 
	const unsigned int &path_num = 32, const float &nak_m = 1, const float &shadow_db = 1, const float &power_avg = 1, 
	const float &delta_omega = 0);