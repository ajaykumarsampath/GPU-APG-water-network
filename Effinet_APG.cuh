/*
 * Effinet_APG.cuh
 * This file has the implemetation of the accelerated proximal gradient
 * method
 *  Created on: Aug 5, 2015
 *      Author: ajay
 *     
 * This file implement the APG algorithm -- equation 26 of the paper.
 * List of functions in this file :
 * 1) __global__ void accelerated_dual_update(T *dual_k,
 *                 T *dual_k_1,T *accelerated_dual,T alpha,int size);
 * 2) __global__ void projection_state(T *x,T *lb,T *ub,T *safety_level,int size);
 * 3) __global__ void projection_control(T *u,T *lb,T *ub,int size);
 * 4) __global__ void dual_update(T *accelerated_dual,T *primal_dual,T *z_dual,
 *                     T *update_dual,float step_size,int size);
 * 5) __host__ void APG_algorithm();
 * 6) __host__ void effinet_proximal_function_g();
 * 7) __host__ void test_effinet_proximal_function_g();
 * 8) __host__ void test_projection_state();
 */

#ifndef EFFINET_APG_CUH_
#define EFFINET_APG_CUH_
#include "Effinet_solve_step.cuh"

template<typename T>__global__ void accelerated_dual_update(T *dual_k,T *dual_k_1,T *accelerated_dual,T alpha,int size){

	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<size){
		real_t temp=(dual_k_1[tid]-dual_k[tid]);
		accelerated_dual[tid]=dual_k_1[tid]+alpha*temp;
		dual_k[tid]=dual_k_1[tid];
	}
}

template<typename T>__global__ void projection_state(T *x,T *lb,T *ub,T *safety_level,int size){

	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int tid_blck=threadIdx.x;
	int tid_box=blockIdx.x*NX;
	if(tid<size){
		if(tid_blck<NX){
			tid_box=tid_box+tid_blck;
			if(x[tid]<lb[tid_box]){
				x[tid]=lb[tid_box];
			}else if(x[tid]>ub[tid_box]){
				x[tid]=ub[tid_box];
			}
		}else{
			tid_box=tid_box+tid_blck-NX;
			if(x[tid]<safety_level[tid_box]){
				x[tid]=safety_level[tid_box];
			}
		}
	}
}

template<typename T>__global__ void projection_control(T *u,T *lb,T *ub,int size){

	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<size){
		if(u[tid]<lb[tid]){
			u[tid]=lb[tid];
		}else if(u[tid]>ub[tid]){
			u[tid]=ub[tid];
		}
	}
}

template<typename T>__global__ void dual_update(T *accelerated_dual,T *primal_dual,T *z_dual,T *update_dual,
		float step_size,int size){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<size){
		update_dual[tid]=accelerated_dual[tid]+step_size*(primal_dual[tid]-z_dual[tid]);
	}
}

__host__ void APG_algorithm(){

	real_t extrapolate_step;
	real_t primal_infes_iter;
	real_t min_al=-1;
	real_t multiple_prd=1;
	int length=(2*NX+NU)*N_NODES;
	real_t inv_neta=0;
	inv_lambda=1/step_size[0];

	for(int i=0;i<iterate[0];i++){
		extrapolate_step=ntheta[1]*(1/ntheta[0]-1);
		/* Equation 26(a)*/
		accelerated_dual_update<real_t><<<N_NODES,2*NX>>>(dev_xi,dev_update_xi,dev_accelarated_xi,extrapolate_step,2*N_NODES*NX);
		accelerated_dual_update<real_t><<<N_NODES,NU>>>(dev_psi,dev_update_psi,dev_accelarated_psi,extrapolate_step,N_NODES*NU);
		//_CUDA(cudaMemset(dev_x,0,NX*N_NODES*sizeof(real_t)));
		/* Equaiton 26(b)*/
		effinet_solve_step();
		/* Equation 26(c)*/
		effinet_proximal_function_g();
		/* Equation 26(d)*/
		dual_update<real_t><<<N_NODES,2*NX>>>(dev_accelarated_xi,dev_primal_xi,dev_dual_xi,dev_update_xi,step_size[0],2*NX*N_NODES);
		dual_update<real_t><<<N_NODES,NU>>>(dev_accelarated_psi,dev_primal_psi,dev_dual_psi,dev_update_psi,step_size[0],NU*N_NODES);
		
		/* Equation 26(e)*/
		ntheta[0]=ntheta[1];
		ntheta[1]=0.5*(sqrt(pow(ntheta[1],4)+4*pow(ntheta[1],2))-pow(ntheta[1],2));
		
	}
	
}


__host__ void effinet_proximal_function_g(){

	/* primal_z^v=Hx^v*/
	_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,2*NX,1,NX,&alpha,(const float**)dev_ptr_F,2*NX,(const float**)dev_ptr_x,
			2*NX,&beta,dev_ptr_primal_xi,2*NX,N_NODES));
	_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NU,1,NU,&alpha,(const float**)dev_ptr_G,NU,(const float**)dev_ptr_u,
			NU,&beta,dev_ptr_primal_psi,NU,N_NODES));

	_CUDA(cudaMemcpy(dev_dual_xi,dev_primal_xi,2*N_NODES*NX*sizeof(real_t),cudaMemcpyDeviceToDevice));
	_CUDA(cudaMemcpy(dev_dual_psi,dev_primal_psi,N_NODES*NU*sizeof(real_t),cudaMemcpyDeviceToDevice));

	_CUBLAS(cublasSaxpy_v2(handle,2*NX*N_NODES,&inv_lambda,dev_accelarated_xi,1,dev_dual_xi,1));
	_CUBLAS(cublasSaxpy_v2(handle,NU*N_NODES,&inv_lambda,dev_accelarated_psi,1,dev_dual_psi,1));

	projection_state<real_t><<<N_NODES,2*NX>>>(dev_dual_xi,dev_xmin,dev_xmax,dev_xs,2*NX*N_NODES);
	projection_control<real_t><<<N_NODES,NU>>>(dev_dual_psi,dev_umin,dev_umax,NU*N_NODES);

}

__host__ void test_effinet_proximal_function_g(){

	real_t **ptr_x_c=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	real_t *y_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));

	real_t* Z_x=(real_t*)malloc(NX*N_NODES*sizeof(real_t*));
	real_t* Z_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));
	real_t* t_x=(real_t*)malloc(2*NX*N_NODES*sizeof(real_t*));
	real_t* t_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));
	real_t* W_x=(real_t*)malloc(2*NX*N_NODES*sizeof(real_t*));
	real_t* W_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));
	inv_lambda=1/step_size[0];
	real_t tol_projection=1;

	char* filepath_Z_x="Data_files_testing/solve_Z_x.h";
	char* filepath_Z_u="Data_files_testing/solve_Z_u.h";
	char* filepath_W_x="Data_files_testing/solve_W_x.h";
	char* filepath_W_u="Data_files_testing/solve_W_u.h";
	char* filepath_t_x="Data_files_testing/solve_t_x.h";
	char* filepath_t_u="Data_files_testing/solve_t_u.h";
	char* filepath_primal_x="Data_files_testing/solve_primal_x.h";
	char* filepath_primal_u="Data_files_testing/solve_primal_u.h";


	allocate_data<real_t>(filepath_Z_x,Z_x);
	allocate_data<real_t>(filepath_Z_u,Z_u);
	allocate_data<real_t>(filepath_W_x,W_x);
	allocate_data<real_t>(filepath_W_u,W_u);
	allocate_data<real_t>(filepath_t_x,t_x);
	allocate_data<real_t>(filepath_t_u,t_u);

	_CUDA(cudaMemcpy(dev_x,Z_x,NX*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_u,Z_u,NU*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_accelarated_xi,W_x,2*NX*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_accelarated_psi,W_u,NU*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));

	effinet_proximal_function_g();

	allocate_data<real_t>(filepath_primal_x,t_x);
	check_correctness_memcpy<real_t>(t_x,dev_primal_xi,2*NX*N_NODES,1,tol_projection);
	allocate_data<real_t>(filepath_t_x,t_x);
	check_correctness_memcpy<real_t>(t_x,dev_dual_xi,2*NX*N_NODES,1,tol_projection);

	allocate_data<real_t>(filepath_primal_u,t_u);
	check_correctness_memcpy<real_t>(t_u,dev_primal_psi,NU*N_NODES,1,tol_projection);
	allocate_data<real_t>(filepath_t_u,t_u);
	check_correctness_memcpy<real_t>(t_u,dev_dual_psi,NU*N_NODES,1,tol_projection);
	printf(" Proximal of g is correct \n");
	/*
	_CUDA(cudaMemcpy(x_c,dev_dual_psi,NU*N_NODES*sizeof(real_t),cudaMemcpyDeviceToHost));
	for(int k=0;k<NU*N_NODES;k++){
		printf("%f %f %d ",x_c[k],t_u[k],k);
	}*/
	free(Z_x);
	free(Z_u);
	free(W_x);
	free(W_u);
	free(t_x);
	free(t_u);
	free(x_c);
	free(y_c);
	free(ptr_x_c);
}

__host__ void test_projection_state(){

	int size=2*NX*N_NODES;
	real_t **ptr_x_c=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	real_t *y_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));

	real_t* t_x=(real_t*)malloc(2*NX*N_NODES*sizeof(real_t*));
	real_t* t_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));

	char* filepath_sum_x="Data_files_testing/solve_sum_x.h";
	char* filepath_sum_u="Data_files_testing/solve_sum_u.h";
	char* filepath_t_x="Data_files_testing/solve_t_x.h";
	char* filepath_t_u="Data_files_testing/solve_t_u.h";

	allocate_data<real_t>(filepath_sum_x,t_x);
	allocate_data<real_t>(filepath_sum_u,t_u);

	_CUDA(cudaMemcpy(dev_dual_xi,t_x,2*NX*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));

	projection_state<real_t><<<N_NODES,2*NX>>>(dev_dual_xi,dev_xmin,dev_xmax,dev_xs,size);

	allocate_data<real_t>(filepath_t_x,t_x);
	allocate_data<real_t>(filepath_t_u,t_u);
	_CUDA(cudaMemcpy(x_c,dev_dual_xi,size*sizeof(real_t),cudaMemcpyDeviceToHost));
	for(int k=0;k<size;k++){
		printf("%f %d ",x_c[k]-t_x[k],k);
	}
	printf("\n");

	free(t_x);
	free(t_u);
	free(x_c);
	free(y_c);
	free(ptr_x_c);
}
#endif /* EFFINET_APG_CUH_ */
