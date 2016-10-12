/*
 * Effinet_data_generation.cuh
 *  Created on: Jul 26, 2015
 *      Author: ajay
 *      
 * This header file has the implementation of the functions that allocate the memory in the
 * GPUs. List of functions: 
 * 1) void create_tree_gpu(void);
 * 2) void create_effinet_gpu(cublasHandle_t handle);
 * 3) void allocate_factor_step_gpu(void);
 * 4) void allocate_solve_step_gpu(void);
 * 5) int allocate_data(char* filepath, T* data)
 * 6) void calculate_particular_solution();
 * 7) void transfer_data_gpu(T *dev_x,T *x,char* filepath,int size,int repeted_time);
 * 8) void free_tree_gpu(void);
 * 9) void free_effinet_gpu(void);
 * 10)void free_factor_step(void);
 * 11)void free_solve_step(void);
 */

#ifndef EFFINET_DATA_GENERATION_CUH_
#define EFFINET_DATA_GENERATION_CUH_
#include "api_effinet_cuda.cuh"
#include "effinet_header.h"


void create_tree_gpu(void){


	_CUDA(cudaMalloc((void**)&DEV_CONST_TREE_STAGES,N_NODES * sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONST_TREE_NODES_PER_STAGE,(N + 1) * sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONST_TREE_LEAVES,K* sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,(N + 2)*sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONSTANT_TREE_NUM_CHILDREN,(N_NONLEAF_NODES)*sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,N_NODES*sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&DEV_CONSTANT_TREE_ANCESTOR,N_NODES*sizeof(uint_t)));

	_CUDA(cudaMemcpy(DEV_CONST_TREE_STAGES,TREE_STAGES,N_NODES * sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONST_TREE_NODES_PER_STAGE,TREE_NODES_PER_STAGE,(N + 1) * sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONST_TREE_LEAVES,TREE_LEAVES,K* sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,TREE_NODES_PER_STAGE_CUMUL,(N + 2)*sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONSTANT_TREE_NUM_CHILDREN,TREE_NUM_CHILDREN,(N_NONLEAF_NODES)*sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,TREE_N_CHILDREN_CUMUL,N_NODES*sizeof(uint_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(DEV_CONSTANT_TREE_ANCESTOR,TREE_ANCESTOR,N_NODES*sizeof(uint_t),cudaMemcpyHostToDevice));
	printf("Tree is allocated in GPUs\n");

}

void create_effinet_gpu(cublasHandle_t handle){
//void create_effinet_gpu(void){
	//real_t prob=1;

	real_t* A;
	real_t* B;
	real_t* F;
	real_t* G;
	real_t* xmin;
	real_t* xmax;
	real_t* xs;

	real_t* u_min;
	real_t* u_max;
	real_t* linear_cost_b;
	real_t* vhat;
	real_t* P_test;
	real_t* L;
	real_t* TREE_VALUE;
	real_t* disturb_w;


	char* filepath_A="Data_files/Effinet_A.h";
	char* filepath_B="Data_files/Effinet_B.h";
	char* filepath_F="Data_files/Effinet_F.h";
	char* filepath_G="Data_files/Effinet_G.h";
	char* filepath_L="Data_files/Effinet_L.h";

	char* filepath_umax="Data_files/Effinet_umax.h";
	char* filepath_umin="Data_files/Effinet_umin.h";
	char* filepath_xmax="Data_files/Effinet_xmax.h";
	char* filepath_xmin="Data_files/Effinet_xmin.h";
	char* filepath_xs="Data_files/Effinet_xs.h";

	//char* filepath_P="Data_files/Effinet_P.h";
	char* filepath_TREE_VALUE="Data_files/Effinet_Tree_value.h";

	/** system dynamics */
	_CUDA(cudaMalloc((void**)&dev_A,K*NX*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_B,K*NX*NU*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_L,K*NU*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_F,2*N_NODES*NX*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_G,N_NODES*NU*NU*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_xmin,N_NODES*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_xmax,N_NODES*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_xs,N_NODES*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_umin,N_NODES*NU*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_umax,N_NODES*NU*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_TREE_VALUE,N_NODES*NX*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_current_state,NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_prev_v,NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_prev_vhat,NU*sizeof(real_t)));

	transfer_data_gpu<real_t>(dev_A,A,filepath_A,NX*NX,K);
	transfer_data_gpu<real_t>(dev_B,B,filepath_B,NX*NU,K);
	transfer_data_gpu<real_t>(dev_L,L,filepath_L,NV*NU,K);
	transfer_data_gpu<real_t>(dev_F,F,filepath_F,2*N_NODES*NX*NX,1);
	transfer_data_gpu<real_t>(dev_G,G,filepath_G,N_NODES*NU*NU,1);


	transfer_data_gpu<real_t>(dev_umin,u_min,filepath_umin,N_NODES*NU,1);
	transfer_data_gpu<real_t>(dev_umax,u_max,filepath_umax,N_NODES*NU,1);
	transfer_data_gpu<real_t>(dev_xmin,xmin,filepath_xmin,N_NODES*NX,1);
	transfer_data_gpu<real_t>(dev_xmax,xmax,filepath_xmax,N_NODES*NX,1);
	transfer_data_gpu<real_t>(dev_xs,xs,filepath_xs,N_NODES*NX,1);

	transfer_data_gpu<real_t>(dev_TREE_VALUE,TREE_VALUE,filepath_TREE_VALUE,N_NODES*NX,1);


	/** cost function*/
	_CUDA(cudaMalloc((void**)&dev_R,N_NODES*NV*NV*sizeof(real_t)));

	/** particular solution*/
	_CUDA(cudaMalloc((void**)&dev_linear_cost_b,N_NODES*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_vhat,N_NODES*NU*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_disturb_w,N_NODES*NX*sizeof(real_t)));
	_CUDA(cudaMemset(dev_linear_cost_b,0,NV*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_vhat,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_disturb_w,0,NX*N_NODES*sizeof(real_t)));

	/** pointers to system dynamics */
	_CUDA(cudaMalloc((void**)&dev_ptr_A,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_B,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_F,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_G,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_L,N_NODES*sizeof(real_t*)));

	_CUDA(cudaMalloc((void**)&dev_ptr_xmin,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_xmax,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_xs,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_umin,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_umax,N_NODES*sizeof(real_t*)));


	_CUDA(cudaMalloc((void**)&dev_ptr_Tree_Value,N_NODES*sizeof(real_t*)));

	/** cost function pointers*/
	_CUDA(cudaMalloc((void**)&dev_ptr_R,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_linear_cost_b,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_vhat,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_disturb_w,N_NODES*sizeof(real_t*)));

	real_t** ptr_A=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_B=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_F=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_G=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_L=(real_t**)malloc(N_NODES*sizeof(real_t*));

	real_t** ptr_xmin=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_xmax=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_xs=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_umin=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_umax=(real_t**)malloc(N_NODES*sizeof(real_t*));


	real_t** ptr_Tree_value=(real_t**)malloc(N_NODES*sizeof(real_t*));

	real_t** ptr_R=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_liner_cost_b=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_vhat=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_disturb_w=(real_t**)malloc(N_NODES*sizeof(real_t*));

	for(int k=0;k<N;k++){
		for(int j=0;j<TREE_NODES_PER_STAGE[k];j++){
			ptr_A[TREE_NODES_PER_STAGE_CUMUL[k]+j]=&dev_A[j*NX*NX];
			ptr_B[TREE_NODES_PER_STAGE_CUMUL[k]+j]=&dev_B[j*NX*NU];
			ptr_L[TREE_NODES_PER_STAGE_CUMUL[k]+j]=&dev_L[j*NU*NV];
		}
	}

	for(int i=0;i<N_NODES;i++){

		ptr_F[i]=&dev_F[i*2*NX*NX];
		ptr_G[i]=&dev_G[i*NU*NU];
		ptr_xmin[i]=&dev_xmin[i*NX];
		ptr_xmax[i]=&dev_xmax[i*NX];
		ptr_xs[i]=&dev_xs[i*NX];
		ptr_umin[i]=&dev_umin[i*NX];
		ptr_umax[i]=&dev_umax[i*NX];

		ptr_Tree_value[i]=&dev_TREE_VALUE[i*NX];

		_CUDA(cudaMemcpy(&dev_R[i*NV*NV],R,NV*NV*sizeof(real_t),cudaMemcpyHostToDevice));
		_CUBLAS(cublasSscal(handle,NV*NV,&TREE_PROB[i],&dev_R[i*NV*NV],1));
		//_CUBLAS(cublasSscal(NU*NU,TREE_PROB[i],&dev_R[i*NU*NU],1));
		ptr_R[i]=&dev_R[i*NV*NV];
		ptr_liner_cost_b[i]=&dev_linear_cost_b[i*NV];
		ptr_vhat[i]=&dev_vhat[i*NU];
		ptr_disturb_w[i]=&dev_disturb_w[i*NX];
	}

	_CUDA(cudaMemcpy(dev_ptr_A,ptr_A,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_B,ptr_B,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_F,ptr_F,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_G,ptr_G,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_L,ptr_L,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_ptr_xmin,ptr_xmin,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_xmax,ptr_xmax,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_xs,ptr_xs,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_umin,ptr_umin,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_umax,ptr_umax,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_ptr_Tree_Value,ptr_Tree_value,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_ptr_R,ptr_R,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_linear_cost_b,ptr_liner_cost_b,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_vhat,ptr_vhat,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_disturb_w,ptr_disturb_w,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	free(ptr_A);
	free(ptr_B);
	free(ptr_F);
	free(ptr_G);
	free(ptr_L);

	free(ptr_xmax);
	free(ptr_xmin);
	free(ptr_xs);
	free(ptr_umin);
	free(ptr_umax);

	free(ptr_Tree_value);
	free(ptr_R);
	free(ptr_liner_cost_b);
	free(ptr_vhat);
	free(ptr_disturb_w);

	printf("Effinet system is allocated in GPUs\n");

}

void allocate_factor_step_gpu(void){

	_CUDA(cudaMalloc((void**)&dev_Effinet_PHI,2*N_NODES*NV*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_PSI,N_NODES*NU*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_THETA,N_NODES*NX*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_OMEGA,N_NODES*NV*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_SIGMA,N_NODES*NV*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_D,2*N_NODES*NV*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_F,N_NODES*NV*NU*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Effinet_G,N_NODES*NV*NX*sizeof(real_t)));

	real_t** ptr_PHI=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_PSI=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_THETA=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_SIGMA=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_OMEGA=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_D=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_F=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_G=(real_t**)malloc(N_NODES*sizeof(real_t*));

	for(int i=0;i<N_NODES;i++){
		ptr_PHI[i]=&dev_Effinet_PHI[2*i*NV*NX];
		ptr_PSI[i]=&dev_Effinet_PSI[i*NV*NU];
		ptr_THETA[i]=&dev_Effinet_THETA[i*NX*NV];
		ptr_OMEGA[i]=&dev_Effinet_OMEGA[i*NV*NV];
		ptr_SIGMA[i]=&dev_Effinet_SIGMA[i*NV];
		ptr_D[i]=&dev_Effinet_D[2*i*NV*NX];
		ptr_F[i]=&dev_Effinet_F[i*NV*NU];
		ptr_G[i]=&dev_Effinet_G[i*NV*NX];
	}

	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_PHI,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_PSI,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_THETA,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_SIGMA,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_OMEGA,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_D,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_F,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Effinet_G,N_NODES*sizeof(real_t*)));

	_CUDA(cudaMemcpy(dev_ptr_Effinet_PHI,ptr_PHI,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_PSI,ptr_PSI,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_THETA,ptr_THETA,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_SIGMA,ptr_SIGMA,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_OMEGA,ptr_OMEGA,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_D,ptr_D,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_F,ptr_F,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Effinet_G,ptr_G,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	/*
	for(int i=TREE_NODES_PER_STAGE_CUMUL[N-1];i<N_NONLEAF_NODES;i++){
		printf("%d %p ", i,ptr_GPAD_K[i]);
	}
	printf("\n");*/
	free(ptr_PHI);
	free(ptr_PSI);
	free(ptr_THETA);
	free(ptr_SIGMA);
	free(ptr_OMEGA);
	free(ptr_D);
	free(ptr_F);
	free(ptr_G);

	printf("Factor step is allocated in GPUs\n");

}

void allocate_solve_step_gpu(void){


	_CUDA(cudaMalloc((void**)&dev_x,NX*(N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_u,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_v,NV*N_NODES*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_xi,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_psi,(NU*N_NODES)*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_update_xi,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_update_psi,NU*N_NODES*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_primal_xi,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_primal_psi,(NU*N_NODES)*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_dual_xi,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_dual_psi,NU*N_NODES*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_accelarated_xi,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_accelarated_psi,NU*N_NODES*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_q,K*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_r,K*NV*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_temp_q,K*NX*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_temp_r,K*NV*sizeof(real_t)));


	_CUDA(cudaMalloc((void**)&dev_ptr_x,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_u,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_v,N_NODES*sizeof(real_t*)));

	_CUDA(cudaMalloc((void**)&dev_ptr_accelerated_xi,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_accelerated_psi,N_NODES*sizeof(real_t*)));

	_CUDA(cudaMalloc((void**)&dev_ptr_primal_xi,N_NODES*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_primal_psi,N_NODES*sizeof(real_t*)));

	_CUDA(cudaMalloc((void**)&dev_ptr_q,K*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_r,K*sizeof(real_t*)));


	real_t** ptr_x=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_u=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_v=(real_t**)malloc(N_NODES*sizeof(real_t*));

	real_t** ptr_accelerated_xi=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_accelerated_psi=(real_t**)malloc(N_NODES*sizeof(real_t*));

	real_t** ptr_primal_xi=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_primal_psi=(real_t**)malloc(N_NODES*sizeof(real_t*));

	real_t** ptr_q=(real_t**)malloc(K*sizeof(real_t*));
	real_t** ptr_r=(real_t**)malloc(K*sizeof(real_t*));

	real_t** ptr_temp_xcst=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t** ptr_temp_ucst=(real_t**)malloc(N_NONLEAF_NODES*sizeof(real_t*));

	for(int i=0;i<N_NODES;i++){
		ptr_x[i]=&dev_x[i*NX];
		ptr_u[i]=&dev_u[i*NU];
		ptr_v[i]=&dev_v[i*NV];

		ptr_accelerated_xi[i]=&dev_accelarated_xi[2*i*NX];
		ptr_accelerated_psi[i]=&dev_accelarated_psi[i*NU];

		ptr_primal_xi[i]=&dev_primal_xi[2*i*NX];
		ptr_primal_psi[i]=&dev_primal_psi[i*NU];

		if(i<K){
			ptr_q[i]=&dev_q[i*NX];
			ptr_r[i]=&dev_r[i*NV];
		}
	}

	_CUDA(cudaMemset(dev_u,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_xi,0,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMemset(dev_psi,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_accelarated_xi,0,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMemset(dev_accelarated_psi,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_update_xi,0,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMemset(dev_update_psi,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_primal_xi,0,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMemset(dev_primal_psi,0,NU*N_NODES*sizeof(real_t)));
	_CUDA(cudaMemset(dev_dual_xi,0,2*(NX*N_NODES)*sizeof(real_t)));
	_CUDA(cudaMemset(dev_dual_psi,0,NU*N_NODES*sizeof(real_t)));

	_CUDA(cudaMemcpy(dev_ptr_x,ptr_x,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_u,ptr_u,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_v,ptr_v,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_accelerated_xi,ptr_accelerated_xi,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_accelerated_psi,ptr_accelerated_psi,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_primal_xi,ptr_primal_xi,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_primal_psi,ptr_primal_psi,N_NODES*sizeof(real_t*),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_ptr_q,ptr_q,K*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_r,ptr_r,K*sizeof(real_t*),cudaMemcpyHostToDevice));


	free(ptr_x);
	free(ptr_u);
	free(ptr_v);
	free(ptr_accelerated_xi);
	free(ptr_accelerated_psi);
	free(ptr_primal_xi);
	free(ptr_primal_psi);

	free(ptr_q);
	free(ptr_r);

	printf("Solve step data is allocated in GPUs\n");

}

template<typename T> int allocate_data(char* filepath, T* data){
	FILE *infile;
	int size;
	infile=fopen(filepath,"r");
	if(infile==NULL){
		printf("%s\n %p", filepath,infile);
		fprintf(stderr,"Error in opening the file %d \n",__LINE__);
		exit(100);
	}else{
		fscanf(infile,"%d \n",&size);
		//printf("Size of the array is %d ",size);
		for(int i=0;i<size;i++){
			fscanf(infile,"%f\n",&data[i]);
		}
		return 0;
	}
}

template<typename T>void calculate_particular_solution(){

	_CUDA(cudaMemcpy(dev_current_state,x,NX*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_prev_v,prev_v,NV*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_prev_vhat,prev_vhat,NU*sizeof(real_t),cudaMemcpyHostToDevice));

	real_t* linear_cost_b=(real_t*)malloc(NV*N_NODES*sizeof(real_t));
	real_t* vhat=(real_t*)malloc(NU*N_NODES*sizeof(real_t));
	real_t* disturb_w=(real_t*)malloc(NX*N_NODES*sizeof(real_t));

	char* filepath_beta="Data_files/Effinet_beta.h";
	char* filepath_vhat="Data_files/Effinet_vhat.h";
	char* filepath_w="Data_files/Effinet_w.h";
	/*
	transfer_data_gpu<real_t>(dev_linear_cost_b,linear_cost_b,filepath_beta,N_NODES*NV,1);
	transfer_data_gpu<real_t>(dev_vhat,vhat,filepath_vhat,N_NODES*NU,1);
	transfer_data_gpu<real_t>(dev_disturb_w,disturb_w,filepath_w,N_NODES*NX,1);
    */
	allocate_data<T>(filepath_beta,linear_cost_b);
	allocate_data<T>(filepath_vhat,vhat);
	allocate_data<T>(filepath_w,disturb_w);

	_CUDA(cudaMemcpy(dev_linear_cost_b,linear_cost_b,N_NODES*NV*sizeof(T),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_vhat,vhat,N_NODES*NU*sizeof(T),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_disturb_w,disturb_w,N_NODES*NX*sizeof(T),cudaMemcpyHostToDevice));

	printf("Particular solutions is calculated for the demand \n");

	free(linear_cost_b);
	free(vhat);
	free(disturb_w);
}

template<typename T>void transfer_data_gpu(T *dev_x,T *x,char* filepath,int size,int repeted_time){

	x=(T*)malloc(size*sizeof(T));
	allocate_data<T>(filepath,x);

	//_CUDA(cudaMalloc((void**)&dev_x,repeted_time*size*sizeof(T)));

	for(int i=0;i<repeted_time;i++){
		_CUDA(cudaMemcpy(&dev_x[i*size],x,size*sizeof(T),cudaMemcpyHostToDevice));
	}

	free(x);

}


void free_tree_gpu(void){

	_CUDA(cudaFree(DEV_CONSTANT_TREE_ANCESTOR));
	_CUDA(cudaFree(DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL));
	_CUDA(cudaFree(DEV_CONSTANT_TREE_NUM_CHILDREN));
	_CUDA(cudaFree(DEV_CONSTANT_TREE_N_CHILDREN_CUMUL));
	_CUDA(cudaFree(DEV_CONST_TREE_CHILDREN));
	_CUDA(cudaFree(DEV_CONST_TREE_LEAVES));
	_CUDA(cudaFree(DEV_CONST_TREE_NODES_PER_STAGE));
	_CUDA(cudaFree(DEV_CONST_TREE_STAGES));
	_CUDA(cudaFree(dev_TREE_VALUE));
	_CUDA(cudaFree(dev_ptr_Tree_Value));

	printf("Tree is deallocated from GPUs\n");
}

void free_effinet_gpu(void){
	_CUDA(cudaFree(dev_A));
	_CUDA(cudaFree(dev_B));
	_CUDA(cudaFree(dev_F));
	_CUDA(cudaFree(dev_G));
	_CUDA(cudaFree(dev_L));
	_CUDA(cudaFree(dev_R));

	_CUDA(cudaFree(dev_xmin));
	_CUDA(cudaFree(dev_xmax));
	_CUDA(cudaFree(dev_xs));
	_CUDA(cudaFree(dev_umin));
	_CUDA(cudaFree(dev_umax));

	_CUDA(cudaFree(dev_linear_cost_b));
	_CUDA(cudaFree(dev_vhat));
	_CUDA(cudaFree(dev_disturb_w));

	_CUDA(cudaFree(dev_ptr_A));
	_CUDA(cudaFree(dev_ptr_B));
	_CUDA(cudaFree(dev_ptr_F));
	_CUDA(cudaFree(dev_ptr_G));
	_CUDA(cudaFree(dev_ptr_L));
	_CUDA(cudaFree(dev_ptr_R));

	_CUDA(cudaFree(dev_ptr_xmin));
	_CUDA(cudaFree(dev_ptr_xmax));
	_CUDA(cudaFree(dev_ptr_xs));
	_CUDA(cudaFree(dev_ptr_umin));
	_CUDA(cudaFree(dev_ptr_umax));

	_CUDA(cudaFree(dev_ptr_linear_cost_b));
	_CUDA(cudaFree(dev_ptr_vhat));
	_CUDA(cudaFree(dev_ptr_disturb_w));

	_CUDA(cudaFree(dev_current_state));
	_CUDA(cudaFree(dev_prev_v));
	_CUDA(cudaFree(dev_prev_vhat));
	printf("Effinet system is deallocated from GPUs\n");
}

void free_factor_step(void){
	_CUDA(cudaFree(dev_Effinet_PHI));
	_CUDA(cudaFree(dev_Effinet_PSI));
	_CUDA(cudaFree(dev_Effinet_THETA));
	_CUDA(cudaFree(dev_Effinet_SIGMA));
	_CUDA(cudaFree(dev_Effinet_OMEGA));
	_CUDA(cudaFree(dev_Effinet_D));
	_CUDA(cudaFree(dev_Effinet_F));
	_CUDA(cudaFree(dev_Effinet_G));

	_CUDA(cudaFree(dev_ptr_Effinet_PHI));
	_CUDA(cudaFree(dev_ptr_Effinet_PSI));
	_CUDA(cudaFree(dev_ptr_Effinet_THETA));
	_CUDA(cudaFree(dev_ptr_Effinet_SIGMA));
	_CUDA(cudaFree(dev_ptr_Effinet_OMEGA));
	_CUDA(cudaFree(dev_ptr_Effinet_D));
	_CUDA(cudaFree(dev_ptr_Effinet_F));
	_CUDA(cudaFree(dev_ptr_Effinet_G));

	printf("Factor step is deallocated from GPUs \n");
}

void free_solve_step(void){

	_CUDA(cudaFree(dev_x));
	_CUDA(cudaFree(dev_u));
	_CUDA(cudaFree(dev_v));
	_CUDA(cudaFree(dev_xi));
	_CUDA(cudaFree(dev_psi));
	_CUDA(cudaFree(dev_update_xi));
	_CUDA(cudaFree(dev_update_psi));
	_CUDA(cudaFree(dev_accelarated_xi));
	_CUDA(cudaFree(dev_accelarated_psi));
	_CUDA(cudaFree(dev_primal_xi));
	_CUDA(cudaFree(dev_primal_psi));
	_CUDA(cudaFree(dev_dual_xi));
	_CUDA(cudaFree(dev_dual_psi));

	_CUDA(cudaFree(dev_temp_q));
	_CUDA(cudaFree(dev_temp_r));

	_CUDA(cudaFree(dev_ptr_x));
	_CUDA(cudaFree(dev_ptr_u));
	_CUDA(cudaFree(dev_ptr_v));
	_CUDA(cudaFree(dev_ptr_accelerated_xi));
	_CUDA(cudaFree(dev_ptr_accelerated_psi));
	_CUDA(cudaFree(dev_ptr_primal_xi));
	_CUDA(cudaFree(dev_ptr_primal_psi));

	printf("Solve step is deallocated from GPUs \n");

}
#endif /* EFFINET_DATA_GENERATION_CUH_ */
