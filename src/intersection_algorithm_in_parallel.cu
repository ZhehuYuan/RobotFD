#include <chrono>
#include <assert.h>
#include <cfloat>
#include <algorithm>

#include "intersection_algorithm_in_parallel.hpp"

unsigned long async;

template<int dims>
__global__ void reachable(
	unsigned int * marker,
	double *global_bound_next_round,
    double *points_curve1_p,
    double *points_curve2_p,
    int curve1_size,
    int curve2_size,
    const double radius2,
	double* dev_result,
	double* intra_block_buffer,
	unsigned int buffer_size
);
Cuda_intersection::Cuda_intersection(const Curve& curve1, const Curve& curve2, bool* forceOversize) {
		cudaError_t cudaStatus;
        point_dimensions = curve1[0].coordinates.size();
        points_curve1_p = (double*)malloc(sizeof(double)*point_dimensions*curve1.size());
        points_curve2_p = (double*)malloc(sizeof(double)*point_dimensions*curve2.size());
		curve1_size = curve1.size();
		curve2_size = curve2.size();
		intervalTime = 0.0;
		reachTime = 0.0;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		n_thread_per_block = prop.maxThreadsPerBlock;
		clockRateHz = prop.clockRate * 1000.0;
		n_block = prop.multiProcessorCount;
		printf("%d threads per block, %d blocks.\n", n_thread_per_block, n_block);
		long memCons = 0;
		size_t bytes;
		size_t free_bytes = 0, total_bytes = 0;
        //Fill the points:
        for( unsigned int i = 0; i < curve1.size(); i++){
        	for( unsigned int j = 0; j < point_dimensions; j++){
                points_curve1_p[i * point_dimensions + j] = curve1[i].coordinates[j];
            }
        }
        for( unsigned int i = 0; i < curve2.size(); i++){
        	for( unsigned int j = 0; j < point_dimensions; j++){
                points_curve2_p[i * point_dimensions + j] = curve2[i].coordinates[j];
            }
        }
        cudaStatus = cudaSetDevice(0);
        if(cudaStatus != cudaSuccess){
            std::cerr << "CUDASetDevice failed! Do you have CUDA-capable GPU installed?" << std::endl;
        }
        
        cudaStatus = cudaMalloc((void**)&dev_points_curve1_p, sizeof(double) * point_dimensions * curve1.size());
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc dev_points_curve1 failed!" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMalloc((void**)&dev_points_curve2_p, sizeof(double) * point_dimensions * curve2.size());
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc dev_points_curve2 failed!" << std::endl;
            goto Error;
        }
		memCons += sizeof(double) * point_dimensions * (curve1.size() + curve2.size());

        cudaStatus = cudaMalloc((void**)&marker, sizeof(unsigned int) * n_block);
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc marker failed!" << std::endl;
            goto Error;
        }
		memCons += sizeof(unsigned int) * n_block;
		cudaStatus = cudaMalloc((void**)&dev_bound_next_round, sizeof(double) * curve2.size());
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc dev_bound_next_round failed!" << std::endl;
            goto Error;
        }
		memCons += sizeof(double) * curve2.size();
		cudaStatus = cudaMalloc((void**)&dev_result, sizeof(double));
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc dev_result failed!" << std::endl;
            goto Error;
        }
		memCons += sizeof(double);
		cudaStatus = cudaMemGetInfo(&free_bytes, &total_bytes);
		if(cudaStatus != cudaSuccess){
            std::cerr << "CudaStatus failed!" << std::endl;
            goto Error;
        }
		if(free_bytes < (long)n_block * (long)n_thread_per_block + (((long)curve1_size - 1) * ((long)curve2_size - 2 + (long)curve1_size - 2) + (long)curve1_size) * (long)sizeof(double) * 4){
			*forceOversize = true;
		}else{
			free_bytes -= (long)n_block * (long)n_thread_per_block + (((long)curve1_size - 1) * ((long)curve2_size - 2 + (long)curve1_size - 2) + (long)curve1_size) * (long)sizeof(double) * 4;
		}	
		async = std::min(free_bytes/(n_block*sizeof(double)), (unsigned long)curve1_size);

		bytes = sizeof(double) * n_block * async;
        cudaStatus = cudaMalloc((void**)&buffer, bytes);
        if(cudaStatus != cudaSuccess){
            std::cerr << "CudaMalloc buffer failed!" << std::endl;
            goto Error;
        }
		memCons += bytes;
		printf("Oversize: %d\n", *forceOversize);	
		if(!*forceOversize){
			long tmp = ((curve1_size - 1) * (curve2_size - 2 + curve1_size - 2) + curve1_size) * sizeof(double);
			cudaStatus = cudaMalloc((void**)&interval_topLB, tmp);
        	if(cudaStatus != cudaSuccess){
        	    std::cerr << "CudaMalloc interval_topLB failed!" << std::endl;
        	    goto Error;
        	}
			cudaStatus = cudaMalloc((void**)&interval_topUB, tmp);
        	if(cudaStatus != cudaSuccess){
        	    std::cerr << "CudaMalloc interval_topUB failed!" << std::endl;
        	    goto Error;
        	}
			cudaStatus = cudaMalloc((void**)&interval_rightLB, tmp);
        	if(cudaStatus != cudaSuccess){
        	    std::cerr << "CudaMalloc interval_rightLB failed!" << std::endl;
        	    goto Error;
        	}
			cudaStatus = cudaMalloc((void**)&interval_rightUB, tmp);
        	if(cudaStatus != cudaSuccess){
        	    std::cerr << "CudaMalloc interval_rightUB failed!" << std::endl;
        	    goto Error;
        	}
			memCons += tmp * 4;
		}

        //Copy Data into device memory
        
        goto NoError;

        Error:
            free_memory();
        NoError:
			printf("Used Memory: %ld\n", memCons);
			printf("Used Memory Excluding Buffer: %ld\n", memCons-bytes);
			return;
}

void Cuda_intersection::data_move(
){
        auto cudaStatus = cudaMemcpy(dev_points_curve1_p, &points_curve1_p[0], sizeof(double)*point_dimensions*curve1_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess ){
            std::cerr << "CudaMemcpy curve1_points to dev_points_curve1_p failed!" << std::endl;
        }
        cudaStatus = cudaMemcpy(dev_points_curve2_p, &points_curve2_p[0], sizeof(double)*point_dimensions*curve2_size, cudaMemcpyHostToDevice );
        if (cudaStatus != cudaSuccess ){
            std::cerr << "CudaMemcpy curve2_points to dev_points_curve2_p failed!" << std::endl;
        }
    cudaStatus = cudaDeviceSynchronize();
    if( cudaStatus != cudaSuccess){
        std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    }

}

void Cuda_intersection::free_memory(){
    cudaFree(dev_points_curve1_p);
    cudaFree(dev_points_curve2_p);
	cudaFree(buffer);
	cudaFree(marker);
	cudaFree(dev_result);
	cudaFree(dev_bound_next_round);
}

__global__ void VEFD_init(
	unsigned int * marker,
	double *global_bound_next_round,
    int curve2_size,
	double* dev_result
){
	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int total_block = gridDim.x;
	const unsigned int total_thread = blockDim.x * gridDim.x;

	int i;

	for(i = thread_id; i < total_block; i += total_thread) marker[i] = 0;
	if(thread_id == 0){
		marker[0] = curve2_size;
		*dev_result = -3.0;
	}

	for(i = thread_id; i < curve2_size; i += total_thread){
		global_bound_next_round[i] = FLT_MAX;
	}
}

template<int dims>
__global__ void VEFD(
	double init_value,
	unsigned int * marker,
	double *global_bound_next_round,
    double *points_curve1_p,
    double *points_curve2_p,
    int curve1_size,
    int curve2_size,
	double* dev_result,
	double* intra_block_buffer,
	unsigned int buffer_size
){
	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int thread_id_inBlock = threadIdx.x;
	const unsigned int block_id = blockIdx.x;
	const unsigned int total_thread = blockDim.x * gridDim.x;
	unsigned int total_block = gridDim.x;
	const unsigned int thread_per_block = blockDim.x;
	if(thread_id - thread_id_inBlock > curve1_size - 2) return;
	total_block -= max(((int)total_thread - curve1_size + 1) / (int)thread_per_block, 0);

	extern __shared__ double smem[];

	double* readBuffer = intra_block_buffer + buffer_size * (block_id - 1);
	double* writeBuffer = intra_block_buffer + buffer_size * block_id;
	double* inter_block_buffer = smem;
	
  	unsigned int round = 0;
	int i;  

  	int curve1_index = thread_id;
  	int curve2_index = -(1 + thread_id_inBlock);

  	double pa1[dims], pa2[dims];
  	double pb1[dims], pb2[dims];
	{
		const unsigned int tmp1 = curve1_index * dims;
		const unsigned int tmp2 = tmp1 + dims;
		for(i = 0; i < dims; i++){
			pa1[i] = points_curve1_p[tmp1 + i];
			pa2[i] = points_curve1_p[tmp2 + i];
		}
	}

  	unsigned int bound_place = 0;
  	int totalRound;
	int tmp = (curve1_size - 1 + thread_per_block - 1) / thread_per_block;
	totalRound = tmp / total_block;
	if(totalRound * total_block + block_id < tmp) totalRound++;
	totalRound *= curve2_size - 1;
	totalRound += thread_per_block - 1;
  	// extra round to clear out inter_buffer
  	totalRound += 2 * thread_per_block;
  
  	unsigned int writeOffset = thread_per_block - 1 - thread_id_inBlock;
  	unsigned int readOffset = writeOffset;
  
	double left = FLT_MAX, bot = FLT_MAX, top, right;
	if(thread_id == 0) bot = init_value * init_value;

  	while(round <= totalRound){
		__syncthreads();
	
		curve2_index++;
		
		if(curve2_index >= curve2_size - 1){
			if(curve1_index + total_thread - thread_id_inBlock < curve1_size - 1){
				curve2_index = min(0, curve2_index - total_thread);
				curve1_index += total_thread;
				bot = FLT_MAX;
				if(curve1_index < curve1_size - 1){
					unsigned int tmp1 = curve1_index * dims;
					unsigned int tmp2 = tmp1 + dims;
					for(i = 0; i < dims; i++){
						pa1[i] = points_curve1_p[tmp1 + i];
						pa2[i] = points_curve1_p[tmp2 + i];
					}
				}
			}
		}
	
		// banch operation
		if((round % thread_per_block) == 0){
			unsigned int inter_location = (round + thread_per_block - 1 - thread_id_inBlock) % (2 * thread_per_block);
			// write to global, unless
			//						1. first 2 * thread_per_block round, nothing to write
			//						2. last large round of last warp, nobody is reading
			if(round >= 2 * thread_per_block && (curve1_index - thread_id_inBlock + thread_per_block < curve1_size - 1 || round + curve2_size < totalRound)){
				if(block_id == total_block - 1){
					// no spinning needed, garentee have space to write
					global_bound_next_round[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();
	
					// ensure all writes are done
					__syncthreads();
				
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[0], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= curve2_size - 1) writeOffset -= curve2_size - 1;
				}else{
					if(thread_id_inBlock == 0){
						while(marker[block_id + 1] > buffer_size - thread_per_block){	
							__threadfence();
						}
					}

					// only one thread (warp) is spining
					__syncthreads();

					writeBuffer[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();

					// ensure all writes are done
					__syncthreads();
				
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[block_id + 1], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= buffer_size) writeOffset -= buffer_size;
				}
			}
			// read from global
			if(curve2_index + thread_id_inBlock < curve2_size - 1 || curve1_index - thread_id_inBlock + total_thread < curve1_size - 1){
				unsigned int tmp = thread_per_block;
				if(curve2_index + total_thread > totalRound){
					tmp = min(tmp, curve2_size - 1 - curve2_index);
				}
				if(thread_id_inBlock == 0){
					while(marker[block_id] < tmp){
						__threadfence();
					}
				}
	
				// only one thread (warp) is spining
				__syncthreads();
		
				if(block_id == 0){
					// read from buffer for next round
					inter_block_buffer[inter_location] = global_bound_next_round[readOffset]; 
					readOffset += thread_per_block;
					if(readOffset >= curve2_size - 1) readOffset -= curve2_size - 1;
				}else{
					// read from buffer for this round
					__threadfence();
					inter_block_buffer[inter_location] = readBuffer[readOffset];
					readOffset += thread_per_block;
					if(readOffset >= buffer_size) readOffset -= buffer_size;
				}
			
				//inter_block_buffer is updated		
				__syncthreads();
			
				if(thread_id_inBlock == 0) atomicAdd(&marker[block_id], -tmp);
			}
		}
	
		round ++;
		if (curve2_index >= 0 && curve1_index < curve1_size - 1 && curve2_index < curve2_size - 1){
			// Read incoming left bounds
			left = inter_block_buffer[bound_place];
			{
				unsigned int tmp1 = curve2_index * dims;
				unsigned int tmp2 = tmp1 + dims;
				for(i = 0; i < dims; i++){
					pb1[i] = points_curve2_p[tmp1 + i];
					pb2[i] = points_curve2_p[tmp2 + i];
				}
			}

			double in = fmin(left, bot);
			//right
			{
				double ab2 = 0.0;
				double ap_dot_ab = 0.0;

				for (i = 0; i < dims; ++i) {
					double ab = pb2[i] - pb1[i];
					double ap = pa2[i] - pb1[i];
					ab2 += ab * ab;
					ap_dot_ab += ap * ab;
    			}
				
				double t = ap_dot_ab / ab2;
    			t = fmin(1.0, fmax(0.0, t));
				right = 0.0;
    			for (i = 0; i < dims; ++i) {
        			double q = t * (pb2[i] - pb1[i]) + pb1[i];
        			double d = pa2[i] - q;
        			right += d * d;
    			}
			}
			//top
			{
				double d[dims];
				double ab2 = 0.0;
				double ap_dot_ab = 0.0;

				for (i = 0; i < dims; ++i) {
					double ab = d[i];
					double ap = pb2[i] - pa1[i];
					ab2 += ab * ab;
					ap_dot_ab += ap * ab;
    			}
				
				double t = ap_dot_ab / ab2;
    			t = fmin(1.0, fmax(0.0, t));
				top = 0.0;
    			for (i = 0; i < dims; ++i) {
        			double q = t * d[i] + pa1[i];
        			double d = pb2[i] - q;
        			top += d * d;
    			}
			}

			top = fmax(top, in);
			right = fmax(right, in);
			
			// write to inter buffer
			bot  = top;
			inter_block_buffer[bound_place] = right;
			bound_place ++;
			if(bound_place == 2 * thread_per_block) bound_place -= 2 * thread_per_block;
			if(curve1_index == curve1_size - 2 && curve2_index == curve2_size - 2){
    	    	*dev_result = sqrt(fmin(top, right));
				assert(top>=0.0);
    		}
		}
	}
}


__global__ void DFD_init(
	unsigned int * marker,
	double *global_bound_next_round,
    int curve2_size,
	double* dev_result
){
	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int total_block = gridDim.x;
	const unsigned int total_thread = blockDim.x * gridDim.x;

	int i;

	for(i = thread_id; i < total_block; i += total_thread) marker[i] = 0;
	if(thread_id == 0){
		marker[0] = curve2_size;
		*dev_result = -3.0;
	}

	for(i = thread_id; i < curve2_size; i += total_thread){
		global_bound_next_round[i] = FLT_MAX;
	}
}

template<int dims>
__global__ void DFD(
	unsigned int * marker,
	double *global_bound_next_round,
    double *points_curve1_p,
    double *points_curve2_p,
    int curve1_size,
    int curve2_size,
	double* dev_result,
	double* intra_block_buffer,
	unsigned int buffer_size
){
	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int thread_id_inBlock = threadIdx.x;
	const unsigned int block_id = blockIdx.x;
	const unsigned int total_thread = blockDim.x * gridDim.x;
	unsigned int total_block = gridDim.x;
	const unsigned int thread_per_block = blockDim.x;
 
	if(thread_id - thread_id_inBlock > curve1_size - 1)return;
	total_block -= max(((int)total_thread - curve1_size) / (int)thread_per_block, 0);

	extern __shared__ double smem[];

	double* readBuffer = intra_block_buffer + buffer_size * (block_id - 1);
	double* writeBuffer = intra_block_buffer + buffer_size * block_id;
	double* inter_block_buffer = smem;
	
  	unsigned int round = 0;
	int i;  

  	int curve1_index = thread_id;
  	int curve2_index = -thread_id_inBlock;

  	double pa[dims];
  	double pb[dims];
	for(i = 0; i < dims; i++){
		pa[i] = points_curve1_p[curve1_index * dims + i];
	}

  	unsigned int bound_place = 0;
  	int totalRound;
	int tmp = (curve1_size + thread_per_block - 1) / thread_per_block;
	totalRound = tmp / total_block;
	if(totalRound * total_block + block_id < tmp) totalRound++;
	totalRound *= curve2_size;
	totalRound += thread_per_block - 1;
  	// extra round to clear out inter_buffer
  	totalRound += 2 * thread_per_block;
 
  	unsigned int writeOffset = thread_per_block - 1 - thread_id_inBlock;
  	unsigned int readOffset = writeOffset;
  
	double left = FLT_MAX, bot, top, right;

  	while(round <= totalRound){
		__syncthreads();

		if(curve2_index >= curve2_size){
			if(curve1_index + total_thread - thread_id_inBlock < curve1_size){
				curve2_index = min(0, curve2_index - total_thread);
				curve1_index += total_thread;
				bot = FLT_MAX;
				if(curve1_index < curve1_size){
					for(i = 0; i < dims; i++){
						pa[i] = points_curve1_p[curve1_index * dims + i];
					}
				}
			}
		}

		// banch operation
		if((round % thread_per_block) == 0){
			unsigned int inter_location = (round + thread_per_block - 1 - thread_id_inBlock) % (2 * thread_per_block);
			// write to global, unless
			//						1. first 2 * thread_per_block round, nothing to write
			//						2. last large round of last warp, nobody is reading
			if(round >= 2 * thread_per_block && (curve1_index - thread_id_inBlock + thread_per_block < curve1_size || round + curve2_size + 1 < totalRound)){
				if(block_id == total_block - 1){
					// no spinning needed, garentee have space to write
					global_bound_next_round[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();
	
					// ensure all writes are done
					__syncthreads();
				
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[0], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= curve2_size) writeOffset -= curve2_size;
				}else{
					if(thread_id_inBlock == 0){
						while(marker[block_id + 1] > buffer_size - thread_per_block){	
							__threadfence();
						}
					}

					// only one thread (warp) is spining
					__syncthreads();

					writeBuffer[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();

					// ensure all writes are done
					__syncthreads();
				
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[block_id + 1], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= buffer_size) writeOffset -= buffer_size;
				}
			}
			// read from global
			if(curve2_index + thread_id_inBlock < curve2_size || curve1_index - thread_id_inBlock + total_thread < curve1_size){
				unsigned int tmp = thread_per_block;
				if(curve2_index + total_thread > totalRound){
					tmp = min(tmp, curve2_size - curve2_index);
				}
				if(thread_id_inBlock == 0){
					while(marker[block_id] < tmp){
						__threadfence();
					}
				}
	
				// only one thread (warp) is spining
				__syncthreads();
		
				if(block_id == 0){
					// read from buffer for next round
					inter_block_buffer[inter_location] = global_bound_next_round[readOffset]; 
					__threadfence();
					readOffset += thread_per_block;
					if(readOffset >= curve2_size) readOffset -= curve2_size;
				}else{
					// read from buffer for this round
					inter_block_buffer[inter_location] = readBuffer[readOffset];
					__threadfence();
					readOffset += thread_per_block;
					if(readOffset >= buffer_size) readOffset -= buffer_size;
				}
			
				//inter_block_buffer is updated		
				__syncthreads();
			
				if(thread_id_inBlock == 0) atomicAdd(&marker[block_id], -tmp);
			}
		}
	
		if (curve2_index >= 0 && curve1_index < curve1_size && curve2_index < curve2_size){
			// Read incoming left bounds
			left = inter_block_buffer[bound_place];
			for(i = 0; i < dims; i++){
				pb[i] = points_curve2_p[curve2_index * dims + i];
			}

			double dist = 0.0;
			for(i = 0; i < dims; i++){
				double t = pa[i] - pb[i];
				dist += t * t;
			}
			if(curve1_index == 0 && curve2_index == 0) {bot = FLT_MAX; left = 0;}
			else if(curve1_index == 0) left = FLT_MAX;
			else if(curve2_index == 0) bot = FLT_MAX;
			
			double in = fmin(left, bot);
			top = fmax(dist, in);
			right = fmin(top, bot);
			
			// write to inter buffer
			inter_block_buffer[bound_place] = right;
			bound_place ++;
			if(bound_place == 2 * thread_per_block) bound_place -= 2 * thread_per_block;
			if(curve1_index == curve1_size - 1 && curve2_index == curve2_size - 1){
				*dev_result = sqrt(top);
				assert(top>=0.0);
    		}
			bot  = top;
		}
		round ++;
		curve2_index++;
	}
}

__global__ void init(
	double *global_bound_next_round,
	double* intra_block_buffer,
	double* dev_result,
    unsigned int curve2_size,
	unsigned int buffer_size,
	unsigned int * marker = nullptr
){
	unsigned int i;
	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int total_thread = blockDim.x * gridDim.x;
  	const unsigned int total_block = gridDim.x;
	for(i = thread_id; i < curve2_size - 1; i += total_thread){
		if(marker != nullptr) global_bound_next_round[i] = -1.0;
		else global_bound_next_round[i] = 2.0;
	}
	for(i = thread_id; i < buffer_size * (total_block - 1); i += total_thread){
		intra_block_buffer[i] = 2.0;
	}
	if(marker != nullptr){
		for(i = thread_id; i < total_block; i += total_thread){
			marker[i] = 0;
		}
    	if(thread_id == 0)marker[0] = curve2_size - 1;
	}
    if(thread_id == 0){ 
		*dev_result = -3.0;
	}
}

template<int dims>
__global__ void interval_calculation(
	double radius2,
	unsigned int curve1_size,
	unsigned int curve2_size,
	double *points_curve1_p,
    double *points_curve2_p,
	double *interval_topLB,
	double *interval_topUB,
	double *interval_rightLB,
	double *interval_rightUB
){
	unsigned int i;

	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  	const unsigned int total_thread = blockDim.x * gridDim.x;
	 unsigned int n_chunk = total_thread / (curve1_size - 1);
	if(total_thread >= (curve1_size - 1) * (curve2_size - 1)) n_chunk = curve2_size - 1;
	if(thread_id >= n_chunk * (curve1_size - 1)) return;
	const unsigned int chunk_id = thread_id / (curve1_size - 1);
	unsigned int chunk_size = (curve2_size - 1) / n_chunk;
	const unsigned int remainder = (curve2_size - 1) - n_chunk * chunk_size;

	unsigned int curve1_index = thread_id % (curve1_size - 1);
	unsigned int curve2_index = chunk_id * chunk_size;
	if(chunk_id < remainder){
		chunk_size += 1;
		curve2_index += chunk_id;
	}else{
		curve2_index += remainder;
	}
	unsigned int interval_id = (curve2_index + curve1_index) * (curve1_size - 1) + curve1_index;
	
	unsigned int round = 0;

	double pa1[dims], pa2[dims], pb1[dims], pb2[dims];
	double dA[dims];
	double Aa = 0.0, inv2Aa;
	for(i = 0; i < dims; i++){
		pa1[i] = points_curve1_p[curve1_index * dims + i       ];
		pa2[i] = points_curve1_p[curve1_index * dims + i + dims];
    	double di = pa2[i] - pa1[i];
		dA[i] = di;
    	Aa += di * di;
	}
	inv2Aa = 0.5 / Aa;


	double topLB, topUB, rightLB, rightUB;

	while(round < chunk_size){
		round ++;
		double Ab = 0.0, inv2Ab;
		{
			rightLB = -1.0;
			rightUB = -2.0;
			double md = 0.0, mm = 0.0;
			for(i = 0; i < dims; i++){
				pb1[i] = points_curve2_p[curve2_index * dims + i       ];
				pb2[i] = points_curve2_p[curve2_index * dims + i + dims];
    			double di = pb2[i] - pb1[i];
    			double mi = pb1[i] - pa2[i];
    			Ab += di * di;
    			md += mi * di;
    			mm += mi * mi;
			}
			inv2Ab = 0.5 / Ab;	
			double Bb = 2.0 * md;
			
			double disc = Bb * Bb - 4.0 * Ab * (mm - radius2);

			if (disc >= 0.0) {
    			double s  = sqrt(disc);

    			double ta = (-Bb - s) * inv2Ab;
    			double tb = (-Bb + s) * inv2Ab;

    			if (!(tb < 0.0 || ta > 1.0)) {

        			if (ta < 0.0) ta = 0.0;
        			if (tb > 1.0) tb = 1.0;

        			rightLB = ta;
        			rightUB = tb;
    			}
			}
		}	
		{
			topLB = -1.0;
			topUB = -2.0;
			double md = 0.0, mm = 0.0;

			for (i = 0; i < dims; i++) {
    			double mi = pa1[i] - pb2[i];
				double di = dA[i];
				md += mi * di;
				mm += mi * mi;
			}

			double Ba = 2.0 * md;
			
			double disc = Ba * Ba - 4.0 * Aa * (mm - radius2);
			if (disc >= 0.0) {
				double s = sqrt(disc);

    			double ta = (-Ba - s) * inv2Aa;
    			double tb = (-Ba + s) * inv2Aa;

    			if (!(tb < 0.0 || ta > 1.0)) {
        			if (ta < 0.0) ta = 0.0;
        			if (tb > 1.0) tb = 1.0;

        			topLB = ta;
        			topUB = tb;
    			}
			}
		}
		interval_topLB[interval_id] = topLB;
		interval_topUB[interval_id] = topUB;
		interval_rightLB[interval_id] = rightLB;
		interval_rightUB[interval_id] = rightUB;
		if(curve1_index == curve1_size -2 && curve2_index == curve2_size - 2) assert(rightUB == 1.0);
		interval_id += curve1_size - 1;
		curve2_index += 1;
	}
}

__global__ void reachability(
	unsigned int * marker,
	unsigned int curve1_size,
	unsigned int curve2_size,
	double *global_bound_next_round,
	double* intra_block_buffer,
	unsigned int buffer_size,
	double* dev_result,
	double *interval_topLB,
	double *interval_topUB,
	double *interval_rightLB,
	double *interval_rightUB
){
  	const unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  	const unsigned int thread_id_inBlock = threadIdx.x;
  	const unsigned int block_id = blockIdx.x;
  	const unsigned int total_thread = blockDim.x * gridDim.x;
  	unsigned int total_block = gridDim.x;
  	const unsigned int thread_per_block = blockDim.x;
	if(thread_id - thread_id_inBlock > curve1_size - 2) return;
	total_block -= (total_thread - curve1_size + 1) / thread_per_block;
 
	extern __shared__ double smem[];
  
	double* readBuffer = intra_block_buffer + buffer_size * (block_id - 1);
  	double* writeBuffer = intra_block_buffer + buffer_size * block_id;
	double* inter_block_buffer = smem;
  	unsigned int writeOffset = thread_per_block - 1 - thread_id_inBlock;
  	unsigned int readOffset = writeOffset;

	double rightUB, rightLB, topLB, topUB, leftLB, botLB;

	int curve1_index = thread_id;
	int curve2_index = -thread_id_inBlock;

	if(curve1_index == 0) botLB = 0.0;
	else botLB = -1.0;

	unsigned int round = 0; 
	unsigned int bound_place = 0;
	int totalRound;
	totalRound = curve2_size - 1 + thread_per_block - 1;
  	// extra round to clear out inter_buffer
  	totalRound += 2 * thread_per_block;
	int interval_id = curve1_index + (curve1_index + curve2_index) * (curve1_size - 1);
	
	while(round < totalRound){
		__syncthreads();
 
		if((round % thread_per_block) == 0){
			unsigned int inter_location = (round + thread_per_block - 1 - thread_id_inBlock) % (2 * thread_per_block);
			// write to global, unless
			//						1. first 2 * thread_per_block round, nothing to write
			//						2. last large round of last warp, nobody is reading
			if(round >= 2 * thread_per_block && (curve1_index - thread_id_inBlock + thread_per_block < curve1_size - 1 || round + curve2_size < round)){
				if(block_id == total_block - 1){
					// no spinning needed, garentee have space to write
					global_bound_next_round[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();
	
					// ensure all writes are done
					__syncthreads();
					
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[0], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= curve2_size - 1) writeOffset -= curve2_size - 1;
				}
				else{
					if(thread_id_inBlock == 0){
						while(marker[block_id + 1] > buffer_size - thread_per_block){	
							__threadfence();
						}
					}

					// only one thread (warp) is spining
					__syncthreads();

					writeBuffer[writeOffset] = inter_block_buffer[inter_location];
					__threadfence();

					// ensure all writes are done
					__syncthreads();
				
					if(thread_id_inBlock == 0){
						atomicAdd(&marker[block_id + 1], thread_per_block);
					}
					writeOffset += thread_per_block;
					if(writeOffset >= buffer_size) writeOffset -= buffer_size;
				}
			}
			
			// Read from global
			if(curve2_index + thread_id_inBlock < curve2_size - 1 || curve1_index - thread_id_inBlock + total_thread < curve1_size - 1){
				unsigned int tmp = thread_per_block;
				if(curve2_index + total_thread > totalRound){
					tmp = min(tmp, curve2_size - 1 - curve2_index);
				}
				if(thread_id_inBlock == 0){
					while(marker[block_id] < tmp){
						__threadfence();
					}
				}
	
				// only one thread (warp) is spining
				__syncthreads();
		
				if(block_id == 0){
					// read from buffer for next round
					inter_block_buffer[inter_location] = global_bound_next_round[readOffset]; 
					readOffset += thread_per_block;
					if(readOffset >= curve2_size - 1) readOffset -= curve2_size - 1;
				}else{
					// read from buffer for this round
					__threadfence();
					inter_block_buffer[inter_location] = readBuffer[readOffset];
					readOffset += thread_per_block;
					if(readOffset >= buffer_size) readOffset -= buffer_size;
				}
				__syncthreads();
				if(thread_id_inBlock == 0) atomicAdd(&marker[block_id], -tmp);
			}
		}
	

		if (curve2_index >= 0 && curve1_index < curve1_size - 1 && curve2_index < curve2_size - 1){
			leftLB  = inter_block_buffer[bound_place]; 
			topLB   = interval_topLB  [interval_id];
			topUB   = interval_topUB  [interval_id];
			rightLB = interval_rightLB[interval_id];
			rightUB = interval_rightUB[interval_id];
			if(botLB < 0.0){
				bool invalid = leftLB > rightUB | leftLB < 0.0;
				double tmp = fmax(rightLB, leftLB);
				rightLB = invalid ? -1.0 : tmp;
				rightUB = invalid ? -2.0 : rightUB;
			}
			if(leftLB < 0.0){
				bool invalid = botLB > topUB | botLB < 0.0;
				double tmp = fmax(topLB, botLB);
				topLB = invalid ? -1.0 : tmp;
				topUB = invalid ? -2.0 : topUB;
			}

			botLB = topLB;
			// write to inter buffer
			inter_block_buffer[bound_place] = rightLB;
			bound_place ++;
			if(bound_place == 2 * thread_per_block) bound_place -= 2 * thread_per_block;
			if(curve1_index == curve1_size - 2 && curve2_index == curve2_size -2){
    		    *dev_result = rightUB;
    		}
		}	
		curve2_index ++;
		round ++;
		interval_id += curve1_size - 1;
	}
}

template<int dims>
__global__ void reachable(
	unsigned int * marker,
	double *global_bound_next_round,
    double *points_curve1_p,
    double *points_curve2_p,
    int curve1_size,
    int curve2_size,
    const double radius2,
	double* dev_result,
	double* intra_block_buffer,
	unsigned int buffer_size
){
  const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int total_thread = blockDim.x * gridDim.x;
  unsigned int total_block = gridDim.x;
  if(thread_id - threadIdx.x > curve1_size - 1)return;
  total_block -= max((total_thread - curve1_size) / (int)blockDim.x, 0);

  extern __shared__ double smem[];

  double* readBuffer = intra_block_buffer + buffer_size * (blockIdx.x - 1);
  double* writeBuffer = intra_block_buffer + buffer_size * blockIdx.x;
  double* inter_block_buffer = smem;
  // (bound + constant * 4) * thread_per_block * 2

  int i, round = 0;
  
  int curve1_index = thread_id;
  int curve2_index = -(1 + threadIdx.x);

  double pa1[dims];
  double pa2[dims];
  double pb2[dims];
  double pb1[dims];

  double Ab, inv2Ab;

  double Aa = 0.0, inv2Aa;

  if(curve1_index < curve1_size){
	int offset1 = curve1_index * dims;
	int offset2 = offset1 + dims;
	for (i = 0; i < dims; ++i) {
		pa1[i] = points_curve1_p[offset1 + i];
		pa2[i] = points_curve1_p[offset2 + i];
    	double di = pa2[i] - pa1[i];
    	Aa += di * di;
	}
	inv2Aa = 0.5 / Aa;
  }

  int totalRound;
  {
	unsigned int tmp = (curve1_size + blockDim.x - 1) / blockDim.x;
	totalRound = tmp / total_block;
	if(totalRound * total_block + blockIdx.x < tmp) totalRound++;
	totalRound *= curve2_size;
	totalRound += blockDim.x - 1;
    // extra round to clear out inter_buffer
    totalRound += 2 * blockDim.x;
  }
  
  int bound_place = 0;

  double leftLB, botLB;
  if(curve1_index == 0) botLB = 0.0;
  else botLB = -1.0;

  int writeOffset = blockDim.x - 1 - threadIdx.x;
  int readOffset = writeOffset;

  while(round <= totalRound){
	// read data	
	
	__syncthreads();
	
	curve2_index++;
	
	if(curve2_index >= curve2_size){
		if(curve1_index + total_thread - threadIdx.x < curve1_size){
			curve2_index = min(0, curve2_index - total_thread);
			curve1_index += total_thread;
			if(curve1_index < curve1_size){
  				Aa = 0.0;
				int offset1  = curve1_index * dims;
				int offset2  = offset1 + dims;
				for (i = 0; i < dims; ++i) {
					pa1[i] = points_curve1_p[offset1 + i];
					pa2[i] = points_curve1_p[offset2 + i];
    				double di = pa2[i] - pa1[i];
    				Aa += di * di;
				}
				inv2Aa = 0.5 / Aa;
 			}
		}
	}

	// banch operation
	if((round % blockDim.x) == 0){
		int inter_location = (round + blockDim.x - 1 - threadIdx.x) % (2 * blockDim.x);
		// write to global, unless
		//						1. first 2 * thread_per_block round, nothing to write
		//						2. last large round of last warp, nobody is reading
		if(round >= 2 * blockDim.x && (curve1_index - threadIdx.x + blockDim.x < curve1_size || round + curve2_size + 1 < totalRound)){
			if(blockIdx.x == total_block - 1u){
				// no spinning needed, garentee have space to write
				global_bound_next_round[writeOffset] = inter_block_buffer[inter_location];
				__threadfence();

				// ensure all writes are done
				__syncthreads();
				
				if(threadIdx.x == 0){
					atomicAdd(&marker[0], blockDim.x);
				}
				writeOffset += blockDim.x;
				if(writeOffset >= curve2_size) writeOffset -= curve2_size;
			}
			else{
				if(threadIdx.x == 0){
					while(marker[blockIdx.x + 1] > buffer_size - blockDim.x){	
						__threadfence();
					}
				}

				// only one thread (warp) is spining
				__syncthreads();

				writeBuffer[writeOffset] = inter_block_buffer[inter_location];
				__threadfence();

				// ensure all writes are done
				__syncthreads();
				
				if(threadIdx.x == 0){
					atomicAdd(&marker[blockIdx.x + 1], blockDim.x);
				}
				writeOffset += blockDim.x;
				if(writeOffset >= buffer_size) writeOffset -= buffer_size;
			}
		}
		// read from global
		if(curve2_index + threadIdx.x < curve2_size || curve1_index - threadIdx.x + total_thread < curve1_size){
			int tmp = blockDim.x;
			if(curve2_index + total_thread > totalRound){
				tmp = min(tmp, curve2_size - curve2_index);
			}
			if(threadIdx.x == 0){
				while(marker[blockIdx.x] < tmp){
					__threadfence();
				}
			}
	
			// only one thread (warp) is spining
			__syncthreads();
		
			if(blockIdx.x == 0){
				// read from buffer for next round
				inter_block_buffer[inter_location] = global_bound_next_round[readOffset]; 
				readOffset += blockDim.x;
				if(readOffset >= curve2_size) readOffset -= curve2_size;
			}else{
				// read from buffer for this round
				__threadfence();
				inter_block_buffer[inter_location] = readBuffer[readOffset];
				readOffset += blockDim.x;
				if(readOffset >= buffer_size) readOffset -= buffer_size;
			}
			int tmp_row = curve2_index + blockDim.x - 1;
			if(tmp_row >= curve2_size) tmp_row -= (curve2_size);
			int offset1 = tmp_row * dims;
			int offset2 = offset1 + dims;
			Ab = 0.0;
			for(i = 0; i < dims; i++){
				pb1[i] = points_curve2_p[offset1 + i];
				pb2[i] = points_curve2_p[offset2 + i];
    			double di = pb1[i] - pb2[i];
    			Ab += di * di;
			}
			inv2Ab = 0.5 / Ab;	
			inter_location += blockDim.x * 2; 
			inter_block_buffer[inter_location] = Ab;
			inter_location += blockDim.x * 2; 
			inter_block_buffer[inter_location] = inv2Ab;
	
			//inter_block_buffer is updated		
			__syncthreads();
			
			if(threadIdx.x == 0) atomicAdd(&marker[blockIdx.x], -tmp);
		}
	}
	
	round ++;
	if (curve2_index >= 0 && curve1_index < curve1_size && curve2_index < curve2_size){
		// Read incoming left bounds
		int tmp = 2 * blockDim.x;
		{
			int offset = bound_place;
			leftLB = inter_block_buffer[offset];
			offset += tmp; 
			Ab    = inter_block_buffer[offset];
			offset += tmp; 
			inv2Ab = inter_block_buffer[offset];
			int offset1 = curve2_index * dims;
			int offset2 = offset1 + dims;
			for(i = 0; i < dims; i++){
				pb1[i] = points_curve2_p[offset1 + i];
				pb2[i] = points_curve2_p[offset2 + i];
			}
		}
		// keep bot bounds
		double topLB, topUB, rightLB, rightUB;
		// right
		{
			rightLB = -1.0;
			rightUB = -2.0;
			double Bb = 0.0, Cb = -radius2;
			for (i = 0; i < dims; ++i) {
    			double di = pb2[i] - pb1[i];
    			double mi = pb1[i] - pa2[i];
    			Bb += mi * di;
    			Cb += mi * mi;
			}
			Bb *= 2.0;
			
			double disc = Bb * Bb - 4.0 * Ab * Cb;

			if (disc >= 0.0) {
    			double s  = sqrt(disc);

    			double ta = (-Bb - s) * inv2Ab;
    			double tb = (-Bb + s) * inv2Ab;

    			// if completely outside [0,1] => no intersection
        		// clamp
    			bool invalid = tb < 0.0 || ta > 1.0;
        		rightLB = invalid ? -1.0 : max(ta, 0.0);
        		rightUB = invalid ? -2.0 : min(tb, 1.0);
			}
		}	
		
		// top
		{
			topLB = -1.0;
			topUB = -2.0;
			double Ba = 0.0, Ca = -radius2;

			for (i = 0; i < dims; i++) {
    			double mi = pa1[i] - pb2[i];
				double di = pa2[i] - pa1[i];
				Ba += mi * di;
				Ca += mi * mi;
			}

			Ba *= 2.0;
			
			double disc = Ba * Ba - 4.0 * Aa * Ca;
			if (disc >= 0.0) {
				double s = sqrt(disc);

    			double ta = (-Ba - s) * inv2Aa;
    			double tb = (-Ba + s) * inv2Aa;

    			// reject if completely outside
        		// clamp
    			bool invalid = tb < 0.0 || ta > 1.0;
        		topLB = invalid ? -1.0 : max(ta, 0.0);
        		topUB = invalid ? -2.0 : min(tb, 1.0);
			}
		}
		// INVALID: test if LB > 1.0 or UB < 0.0?
		if(botLB < 0.0){
			bool invalid = leftLB > rightUB || leftLB < 0.0;
			double tmp2 = fmax(rightLB, leftLB);
			rightLB = invalid ? -1.0 : tmp2;
			rightUB = invalid ? -2.0 : rightUB;
		}
		if(leftLB < 0.0){
			bool invalid = botLB > topUB || botLB < 0.0;
			double tmp2 = fmax(topLB, botLB);
			topLB = invalid ? -1.0 : tmp2;
			topUB = invalid ? -2.0 : topUB;
		}
		botLB = topLB;
		if (curve2_index >= 0 && curve1_index < curve1_size){
			// write to inter buffer
			inter_block_buffer[bound_place] = rightLB;
			bound_place ++;
			if(bound_place == tmp) bound_place -= tmp;
		}
		if(curve1_index == curve1_size - 1 && curve2_index == curve2_size -1){
    	    *dev_result = rightUB;
    	}
	}
  }
}

void Cuda_intersection::Bound_call_gpu(
	double* dfd,
	double* vefd
){
	using namespace std::chrono;

	unsigned int buffer_size = async;
    
	//Cuda launching utils
    cudaError_t cudaStatus;
	DFD_init<<<n_block, n_thread_per_block>>>(
		marker,
		dev_bound_next_round,
    	curve2_size,
		dev_result
	);
    cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Init DFD launch error: %s\n", cudaGetErrorString(err));
    }else{
	#if debug
		printf("Init DFD done\n");
    #endif
	}
	if(point_dimensions == 3){
		DFD<3><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
			marker,
			dev_bound_next_round,
    		dev_points_curve1_p,
    		dev_points_curve2_p,
    		curve1_size,
    		curve2_size,
			dev_result,
			buffer,
			buffer_size
		);
	}else if(point_dimensions == 2){
        DFD<2><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
            marker,
            dev_bound_next_round,
            dev_points_curve1_p,
            dev_points_curve2_p,
            curve1_size,
            curve2_size,
            dev_result,
            buffer,
            buffer_size
        );
	}
    cudaDeviceSynchronize();
	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("DFD launch error: %s\n", cudaGetErrorString(err));
    }else{
	#if debug
		printf("DFD done\n");
    #endif
	}
	
	double result;
    cudaStatus = cudaMemcpy(
        &result, dev_result, sizeof(double), cudaMemcpyDeviceToHost
    );
	if (cudaStatus != cudaSuccess) {
    	printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
	}
    
	*dfd = result;
	
	VEFD_init<<<n_block, n_thread_per_block>>>(
		marker,
		dev_bound_next_round,
    	curve2_size,
		dev_result
	);
    cudaDeviceSynchronize();
	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Init VEFD launch error: %s\n", cudaGetErrorString(err));
    }else{
	#if debug
		printf("Init VEFD done\n");
    #endif
	}
	if(point_dimensions == 3){
		VEFD<3><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
			*vefd,
			marker,
			dev_bound_next_round,
    		dev_points_curve1_p,
    		dev_points_curve2_p,
    		curve1_size,
    		curve2_size,
			dev_result,
			buffer,
			buffer_size
		);
	}else if(point_dimensions == 2){
        VEFD<2><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
            *vefd,
			marker,
            dev_bound_next_round,
            dev_points_curve1_p,
            dev_points_curve2_p,
            curve1_size,
            curve2_size,
            dev_result,
            buffer,
            buffer_size
        );
	}
	
    cudaDeviceSynchronize();
	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("VEFD launch error: %s\n", cudaGetErrorString(err));
    }else{
	#if debug
		printf("VEFD done\n");
	#endif
    }
	
    cudaStatus = cudaMemcpy(
        &result, dev_result, sizeof(double), cudaMemcpyDeviceToHost
    );
	if (cudaStatus != cudaSuccess) {
    	printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
	}
    
	cudaDeviceSynchronize();

	*vefd = result;

	return;
}


bool Cuda_intersection::intersection_interval_call_gpu(
    double radius,
	bool forceOversize
){
	using namespace std::chrono;

    //Cuda launching utils
    cudaError_t cudaStatus;

	unsigned int buffer_size = async;
	init<<<n_block, n_thread_per_block>>>(
		dev_bound_next_round,
		buffer,
       	dev_result,
    	curve2_size,
		buffer_size,
		marker
	);
    cudaDeviceSynchronize();
	cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
    	printf("Init launch error: %s\n", cudaGetErrorString(err2));
    }else{
	#if debug
		printf("Init done\n");
	#endif
	}
    cudaStatus = cudaDeviceSynchronize();
    if( cudaStatus != cudaSuccess){
        std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    }
	auto start = std::chrono::high_resolution_clock::now();
	if(point_dimensions == 3) {
		if(forceOversize){
			start = std::chrono::high_resolution_clock::now();
			reachable<3><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
		    	marker,
				dev_bound_next_round,
    	    	dev_points_curve1_p,
    	    	dev_points_curve2_p,
    	    	curve1_size - 1,
    	    	curve2_size - 1,
    	    	radius * radius,
    	    	dev_result,
    			buffer,
    			buffer_size
    		);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
        		printf("Decider launch error: %s\n", cudaGetErrorString(err));
    		}
		}else{
			start = std::chrono::high_resolution_clock::now();
			interval_calculation<3><<<n_block, n_thread_per_block/2>>>( 
				radius * radius,
				curve1_size,
				curve2_size,
				dev_points_curve1_p,
    			dev_points_curve2_p,
				interval_topLB,
				interval_topUB,
				interval_rightLB,
				interval_rightUB
			);
			cudaError_t err = cudaGetLastError();
    		if (err != cudaSuccess) {
        		printf("Interval launch error: %s\n", cudaGetErrorString(err));
    		}else{
    		}
    		cudaStatus = cudaDeviceSynchronize();
    		if( cudaStatus != cudaSuccess){
        		std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        		std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    		}
			#if debug
				printf("Interval done\n");
			#endif
			auto end = std::chrono::high_resolution_clock::now();
            intervalTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			start = std::chrono::high_resolution_clock::now();
			reachability<<<n_block, (curve1_size + n_block - 1)/n_block, n_thread_per_block * 6 * sizeof(double)>>>(
				marker,
				curve1_size,
				curve2_size,
				dev_bound_next_round,
				buffer,
				buffer_size,
				dev_result,
				interval_topLB,
				interval_topUB,
				interval_rightLB,
				interval_rightUB
			);
			err = cudaGetLastError();
    		if (err != cudaSuccess) {
        		printf("Reachability launch error: %s\n", cudaGetErrorString(err));
    		}else{
			#if debug
				printf("Reachability done\n");
			#endif
    		}
		}
	}else if(point_dimensions == 2) {
		if(forceOversize){
			reachable<2><<<n_block, min((curve1_size + n_block - 1)/n_block, n_thread_per_block), n_thread_per_block * 6 * sizeof(double)>>>(
				marker,
    	    	dev_bound_next_round,
    	    	dev_points_curve1_p,
    	    	dev_points_curve2_p,
    	    	curve1_size - 1,
    	    	curve2_size - 1,
    	    	radius * radius,
    	    	dev_result,
    			buffer,
    			buffer_size
    		);
			cudaError_t err = cudaGetLastError();
    		if (err != cudaSuccess) {
        		printf("Decider launch error: %s\n", cudaGetErrorString(err));
    		}
		}else{
			auto start = std::chrono::high_resolution_clock::now();
			interval_calculation<2><<<n_block, n_thread_per_block>>>(
				radius * radius,
				curve1_size,
				curve2_size,
				dev_points_curve1_p,
    			dev_points_curve2_p,
				interval_topLB,
				interval_topUB,
				interval_rightLB,
				interval_rightUB
			);
			cudaError_t err = cudaGetLastError();
    		if (err != cudaSuccess) {
        		printf("Interval launch error: %s\n", cudaGetErrorString(err));
    		}else{
    		}
    		cudaStatus = cudaDeviceSynchronize();
    		if( cudaStatus != cudaSuccess){
        		std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        		std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    		}
			#if debug
				printf("Interval done\n");
			#endif
			auto end = std::chrono::high_resolution_clock::now();
            intervalTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			reachability<<<n_block, (curve1_size + n_block - 1)/n_block, n_thread_per_block * 6 * sizeof(double)>>>(
				marker,
				curve1_size,
				curve2_size,
				dev_bound_next_round,
				buffer,
				buffer_size,
				dev_result,
				interval_topLB,
				interval_topUB,
				interval_rightLB,
				interval_rightUB
			);
			err = cudaGetLastError();
    		if (err != cudaSuccess) {
        		printf("Reachability launch error: %s\n", cudaGetErrorString(err));
    		}else{
			#if debug
				printf("Reachability done\n");
			#endif
    		}
		}
	}else std::cerr << "Invalid dimension." << std::endl;
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){
        std::cerr << "CudaGetLastError returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Waits until all threads are done with their job.
    cudaStatus = cudaDeviceSynchronize();
    if( cudaStatus != cudaSuccess){
        std::cerr << "CudaDeviceSynchronize() returned error code: " << cudaStatus << " after launching kernel!" << std::endl;
        std::cerr << cudaGetErrorString(cudaStatus) << std::endl;
    }
	auto end = std::chrono::high_resolution_clock::now();
    reachTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
	double result;
    cudaStatus = cudaMemcpy(
        &result, dev_result, sizeof(double), cudaMemcpyDeviceToHost
    );
    if(cudaStatus != cudaSuccess){
        std::cerr << "CudaMemcpy dev_results into results failed!" << std::endl;
    }

	bool ret = result == 1.0;
#if debug
	printf("%.14f, %f %d\n", result, ret);
#endif
    return ret;
}
