#pragma once

#include "geometry_basics.hpp"
#include "curve.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define debug 0

class Cuda_intersection{
    public:
		double intervalTime;
		double reachTime;
        Cuda_intersection(const Curve& curve1, const Curve& curve2, bool* forceOversize);

        void intersection_interval_cuda(
            double radius
        ); 
        bool intersection_interval_call_gpu( 
            double radius,
			bool forceOversize
        );
        
		void Bound_call_gpu(
			double* dfd,
			double* vefd
		);
		void data_move();
        void free_memory(); 
    
    private:

		int curve1_size;
		int curve2_size;
		bool first;

        double * dev_points_curve1_p;
        double * dev_points_curve2_p;
        double * dev_bound_this_round1;
        double * dev_bound_this_round2;
        double * dev_bound_next_round;
        double * dev_const_curve1;
        double * dev_const_curve2;
		double * interval_topLB;
		double * interval_topUB;
		double * interval_rightLB;
		double * interval_rightUB;
		double * dev_result;
		double * buffer;
		unsigned long long int * global_counter;
		unsigned int * marker;
		double clockRateHz;

        unsigned int point_dimensions;
		int n_block;
		int n_thread_per_block;
		double lb2;

        double *points_curve1_p;
        double *points_curve2_p;
};
