#include "SoSGeneWithCuda.h"
#include <iostream>
#include <ctime>
#include <string>

enum FadingType { gaussian, gaussianIQ, lognormal, nakagami, nakagami2, nakagamiIQ, lognak, lognakIQ };

int main()
{
	FadingType fading_type = lognak;

	switch (fading_type){
	case gaussian:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 1000;
		const unsigned int path_num = 32;
		const float mean = 0;
		const float variance = 1;
		const float delta_omega = 0;

		float *arr = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!gaussianGene(arr, length, fs, fd_max, time_spend, path_num, mean, variance, delta_omega) && arr != NULL){
			std::cerr << "gaussianGene调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "gaussianGene调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("gaussian.bin", "wb");
		if (fp){
			fwrite(arr, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case gaussianIQ:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 1000;
		const unsigned int path_num = 32;
		const float delta_omega = 0;

		float *arr_i = NULL;
		float *arr_q = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!gaussianGeneIQ(arr_i, arr_q, length, fs, fd_max, time_spend, path_num, delta_omega) && arr_i != NULL && arr_q != NULL){
			std::cerr << "gaussianGene2调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "gaussianGene2调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("gaussianIQ.bin", "wb");
		if (fp){
			fwrite(arr_i, sizeof(float), length, fp);
			fwrite(arr_q, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case lognormal:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 1000;
		const unsigned int path_num = 32;
		const float mean = 0;
		const float variance = 1;
		const float delta_omega = 0;

		float *arr = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!lognormalGene(arr, length, fs, fd_max, time_spend, path_num, mean, variance, delta_omega) && arr != NULL){
			std::cerr << "lognormalGene调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "lognormalGene调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("lognormal.bin", "wb");
		if (fp){
			fwrite(arr, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case nakagami:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 100;
		const unsigned int path_num = 32;
		const float nak_m = 4;
		const float nak_omega = 1;
		const float delta_omega = 0;

		float *arr = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!nakagamiGene(arr, length, fs, fd_max, time_spend, path_num, nak_m, nak_omega, delta_omega) && arr != NULL){
			std::cerr << "nakagamiGene调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "nakagamiGene调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("nakagami.bin", "wb");
		if (fp){
			fwrite(arr, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case nakagami2:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 100;
		const unsigned int path_num = 32;
		const float nak_m = 1;
		const float nak_omega = 1;
		const float delta_omega = 0;

		float *arr = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!nakagamiGene2(arr, length, fs, fd_max, time_spend, path_num, nak_m, nak_omega, delta_omega) && arr != NULL){
			std::cerr << "nakagamiGene2调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "nakagamiGene2调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("nakagami.bin", "wb");
		if (fp){
			fwrite(arr, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case nakagamiIQ:{
		const float fs = 1000;
		const float fd_max = 50;
		const float time_spend = 100;
		const unsigned int path_num = 32;
		const float nak_m = 1;
		const float nak_omega = 1;
		const float delta_omega = 0;

		float *arr_i = NULL;
		float *arr_q = NULL;
		size_t length;
		clock_t start, stop;

		start = clock();
		if (!nakagamiGeneIQ(arr_i, arr_q, length, fs, fd_max, time_spend, path_num, nak_m, nak_omega, delta_omega) && arr_i != NULL && arr_q != NULL){
			std::cerr << "nakagamiGeneIQ调用失败！" << std::endl;
			break;
		}
		stop = clock();

		std::cout << "nakagamiGeneIQ调用成功！" << std::endl;
		std::cout << "所花时间：" << stop - start << "ms" << std::endl;

		FILE *fp = fopen("nakagamiIQ.bin", "wb");
		if (fp){
			fwrite(arr_i, sizeof(float), length, fp);
			fwrite(arr_q, sizeof(float), length, fp);
			fclose(fp);
		}
		break;
	}
	case lognak:{
		char filename[_MAX_PATH];

		const float fs = 1000;
		const float fd_max = 48.4;
		const float time_spend = 100;
		const unsigned int path_num = 128;
		const float nak_m[] = { 11.3, 10.3, 9.2, 9.6 }; 
		const float shadow_db[] = { 1.1, 3.2, 3.9, 4.2 };
		//const float nak_m[] = { 1, 1, 1, 1 };
		//const float shadow_db[] = { 3, 3, 3, 3 };
		const float power_avg = 1;
		const float delta_omega = 0;

		float *arr = NULL;
		size_t length;
		clock_t start, stop;

		for (int jj = 0; jj < 4; jj++){
			for (int ii = 1; ii <= 1000; ii++){
				start = clock();
				if (!lognakGene(arr, length, fs, fd_max, time_spend, path_num, nak_m[jj], shadow_db[jj], power_avg, delta_omega) && arr != NULL){
					std::cerr << "lognakGene调用失败！" << std::endl;
					break;
				}
				stop = clock();

				std::cout << "lognakGene调用成功！" << std::endl;
				std::cout << "所花时间：" << stop - start << "ms" << std::endl;

				sprintf(filename, "lognak%d_%d.bin", jj, ii);
				FILE *fp = fopen(filename, "wb");
				if (fp){
					fwrite(arr, sizeof(float), length, fp);
					fclose(fp);
				}
			}
		}
		break;
	}
	case lognakIQ:{
	    char filename[_MAX_PATH];

	    const float fs = 1000;
		const float fd_max = 48.4;
		const float time_spend = 100;
		const unsigned int path_num = 128;
		const float nak_m[] = { 11.3, 10.3, 9.2, 9.6 };
		const float shadow_db[] = { 1.1, 3.2, 3.9, 4.2 };
		const float power_avg = 1;
		const float delta_omega = 0;

		float *arr_i = NULL;
		float *arr_q = NULL;
		size_t length;
		clock_t start, stop;

		for (int jj = 0; jj < 4; jj++){
			for (int ii = 1; ii <= 100; ii++){
				start = clock();
				if (!lognakGeneIQ(arr_i, arr_q, length, fs, fd_max, time_spend, path_num, nak_m[jj], shadow_db[jj], power_avg, delta_omega) && arr_i != NULL && arr_q != NULL){
					std::cerr << "lognakGeneIQ调用失败！" << std::endl;
					break;
				}
				stop = clock();

				std::cout << "lognakGene调用成功！" << std::endl;
				std::cout << "所花时间：" << stop - start << "ms" << std::endl;

				sprintf(filename, "lognakIQ%d_%d.bin", jj, ii);
				FILE *fp = fopen(filename, "wb");
				if (fp){
					fwrite(arr_i, sizeof(float), length, fp);
					fwrite(arr_q, sizeof(float), length, fp);
					fclose(fp);
				}
			}
		}
		break;
	}
	}

	std::cin.get();
	return 0;
}
