#include <vector>
#include <limits>

#include <chrono>
#include <iostream>
#include <cassert>
#include <boost/chrono/include.hpp>

#include "frechet.hpp"
#include "intersection_algorithm_in_parallel.hpp"

using namespace std::chrono;

namespace Frechet {

namespace Continuous {

double distance_cuda(const Curve &curve1, const Curve &curve2, const double eps, bool forceOversize){
	double result;

	printf("%ld, %ld\n", curve1.size(), curve2.size());
	
	double lb;
    double ub;
	int iter = 0;
	Cuda_intersection cuda = Cuda_intersection(curve1, curve2, &forceOversize);
	auto start = std::chrono::high_resolution_clock::now();
	cuda.data_move();
	auto end0 = std::chrono::high_resolution_clock::now();
    lb = std::sqrt(std::max(curve1.front().dist_sqr(curve2.front()), curve1.back().dist_sqr(curve2.back())));
	printf("Naive LB: %.14f\n", lb);
	cuda.Bound_call_gpu(&ub, &lb);
	auto end1 = std::chrono::high_resolution_clock::now();
	printf("LB: %.14f\nUB: %.14f\n", lb, ub);
	assert(lb <= ub);
	double split = (ub + lb)/2.0;
	
	{
		if (ub - lb > eps) {
			//Binary search over the feasible distances
			while (ub - lb > eps) {	
				split = (ub + lb)/2.0;
			#if debug
				auto start2 = std::chrono::high_resolution_clock::now();
				printf("\tSplit: %f\n", split);
			#endif
				iter ++;
				auto isLessThan = cuda.intersection_interval_call_gpu(split);
				if (isLessThan) {
					ub = split;
				}
				else {
					lb = split;
				}
			#if debug
				printf("\tTime: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end3 - start2).count() / 1000.0);
				printf("\[%.14f, %.14f\]\n", lb, ub);
				printf("%.14f > %.14f\n", ub - lb, eps);
			#endif
			}
		}
	}
	auto value = (ub + lb)/2.0;
	auto end2 = std::chrono::high_resolution_clock::now();
	printf("Iteration: %d\n", iter);
	printf("Result: %.7f\n", value);
	printf("To GPU Time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end0 - start).count() / 1000.0);
	printf("Pre Time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end1 - end0).count() / 1000.0);
	printf("Interval Time: %f ms\n", cuda.intervalTime);
	printf("Reachability Time: %f ms\n", cuda.reachTime);
	printf("Total Time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end2 - start).count() / 1000.0);
	result = value;
	cuda.free_memory();
	return result;
}

}
}
