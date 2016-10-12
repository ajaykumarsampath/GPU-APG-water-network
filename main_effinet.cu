/*
 * This file has the main function of the accelerated proximal gradient
 * (APG) algorithm for EFFINET system. The theoritical discussion of this 
 * algorithm is given in the paper. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_device_runtime_api.h>
#include <math.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cuda_timer.cuh"
#include "api_effinet_cuda.cuh"
#include "Effinet_APG.cuh"


int main(void){
	/* state and control variables calculated with GUROBI 
	 * used for comparing the APG output*/
	real_t* Z_x=(real_t*)malloc(NX*N_NODES*sizeof(real_t*));
	real_t* Z_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	/* time and tolerance in the comparision*/
	real_t t,tol_apg;
	
	
	char* filepath_x="Data_files/Effinet_x_yalmip.h";
	char* filepath_u="Data_files/Effinet_u_yalmip.h";

	_CUBLAS(cublasCreate(&handle));
	start_tictoc();
	/* allocate the system varaibles in GPU*/
	create_effinet_gpu(handle);
	/* allocate the tree variables in CPU*/
	create_tree_gpu();
	/* allocate the new control variable and the cost function by 
	 * eliminating the control-disturbance constraint -- Appendix A in the paper*/
	calculate_particular_solution<real_t>();
	
	/* allocate memory for factor step matrices in GPU*/
	allocate_factor_step_gpu();
	
	/* allocate memory for solve step matrices in GPU*/
	allocate_solve_step_gpu();
	/* Implementation of the factor step*/
	effinet_factor_step();

	//test_factor_step();
	//test_solve_step();
	//test_effinet_proximal_function_g();
	//test_projection_state();

	tic();
	/* Implements the APG algorithm -- equation 26 in the paper*/
	APG_algorithm();
	t=toc();
	printf("Time to run %d iterations %f \n",iterate[0]+1,t);

	/* GUROBI output*/
	allocate_data<real_t>(filepath_x,Z_x);
	allocate_data<real_t>(filepath_u,Z_u);

	/*
	tol_apg=1e-2;
	check_correctness_memcpy<real_t>(Z_u,dev_u,NU*N_NODES,1,tol_apg);
	tol_apg=20;
	check_correctness_memcpy<real_t>(Z_x,dev_x,NX*N_NODES,1,tol_apg);
	*/
	uint_t size=1;
	_CUDA(cudaMemcpy(x_c,dev_u,NU*size*sizeof(real_t),cudaMemcpyDeviceToHost));
	printf("control \n");
	for(int k=0;k<size;k++){
		for(int j=0;j<NU;j++){
			printf("%f %f ",x_c[k*NU+j]-Z_u[k*NU+j],Z_u[k*NU+j]);
		}
		printf("\n");
	}
	printf("state :\n");
	_CUDA(cudaMemcpy(x_c,dev_x,NX*size*sizeof(real_t),cudaMemcpyDeviceToHost));
	for(int k=0;k<size;k++){
		for(int j=0;j<NX;j++){
			printf("%f %f ",x_c[k*NX+j]-Z_x[k*NX+j],Z_x[k*NX+j]);
		}
		printf("\n");
	}

	//test_summation_children(1,2);
	_CUBLAS(cublasDestroy(handle));
	/* Free the memory -- first solve step, factor step,
	 * tree memory, system dynamics data*/
	free_solve_step();
	free_factor_step();
	free_tree_gpu();
	free_effinet_gpu();
	printf("Exit Effinet running in GPUs \n");
	return(0);
}
