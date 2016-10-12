/*
 * Effinet_solve_step.cuh
 *
 *  Created on: Aug 2, 2015
 *      Author: ajay
 */
/* 
 * This file has the implementation of the solve step for the 
 * EFFINET network -- Algorithm 1 in the paper. 
 * The list of function in this file are 
 * 1) __global__ void summation_children(T *x,T *y,uint_t* DEV_CONST_TREE_NODES_PER_STAGE
		,uint_t* DEV_CONSTANT_TREE_NUM_CHILDREN,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL
		,uint_t* DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,uint_t dim,uint_t stage);
   2) __global__ void child_nodes_update(T *x,T *y,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,
		uint_t* DEV_CONSTANT_TREE_ANCESTOR,int dim,int stage);
   3) void effinet_solve_step(void);
   4) void test_summation_children(int stage,int dim);
   5) void test_multiplication_control();
   6) void test_multiplication_cost();
   7) void test_solve_step();
 */
#ifndef EFFINET_SOLVE_STEP_CUH_
#define EFFINET_SOLVE_STEP_CUH_
#include "Effinet_factor_step.cuh"

template<typename T> __global__ void summation_children(T *x,T *y,uint_t* DEV_CONST_TREE_NODES_PER_STAGE
		,uint_t* DEV_CONSTANT_TREE_NUM_CHILDREN,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL
		,uint_t* DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,uint_t dim,uint_t stage){

	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int relative_node=tid/dim;
	int relative_parent_node=tid-relative_node*dim;
	int no_nodes=DEV_CONST_TREE_NODES_PER_STAGE[stage];
	int node_before=0;
	int off_set=0;
	if(tid<no_nodes*dim){
		if(stage>0){
			node_before=DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL[stage];
			off_set=(DEV_CONSTANT_TREE_N_CHILDREN_CUMUL[node_before+relative_node-1]-DEV_CONSTANT_TREE_N_CHILDREN_CUMUL[node_before-1])*dim;
		}
		int no_child=DEV_CONSTANT_TREE_NUM_CHILDREN[node_before+relative_node];
		if(no_child>1){
			for(int i=0;i<no_child-1;i++){
				if(i==0)
					y[tid]=x[off_set+relative_parent_node]+x[off_set+relative_parent_node+dim];
				if(i>0)
					y[tid]=y[tid]+x[off_set+relative_parent_node+(i+1)*dim];
			}
		}else{
			//printf("%d %d %d %f \n",no_child,dim,relative_node,relative_parent_node,x[tid]);
			y[tid]=x[off_set+relative_parent_node];
		}
	}
}

template<typename T>__global__ void child_nodes_update(T *x,T *y,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,
		uint_t* DEV_CONSTANT_TREE_ANCESTOR,int dim,int stage){

	int tid =blockDim.x*blockIdx.x+threadIdx.x;
	int relative_node=tid/dim;
	int dim_element=tid-relative_node*dim;
	int node_before=DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL[stage+1];
	int pre_ancestor=DEV_CONSTANT_TREE_ANCESTOR[node_before];
	int ancestor=DEV_CONSTANT_TREE_ANCESTOR[node_before+relative_node];
	y[tid]=x[(ancestor-pre_ancestor)*dim+dim_element]+y[tid];
	//y[(ancestor-1)*dim+tid]=y[(ancestor-1)*dim+tid]+x[bid];

}


void effinet_solve_step(void){

	real_t scale[2]={-0.5,1};

	/*real_t tol_solve_step=5e-2;
	real_t **ptr_x_c=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	real_t *y_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));

	char* filepath_v_forward="Data_files_testing/solve_v_forward.h";

	//allocate_data<real_t>(filepath_v_forward,y_c);
	/* r=beta */
	_CUDA(cudaMemcpy(dev_Effinet_SIGMA,dev_linear_cost_b,NV*N_NODES*sizeof(real_t),cudaMemcpyDeviceToDevice));

	for(int k=N-1;k>-1;k--){

		if(k<N-1){
			/* sigma=sigma+r */
			_CUBLAS(cublasSaxpy(handle,TREE_NODES_PER_STAGE[k]*NV,&alpha,dev_r,1,&dev_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]*NV],1));
		}

		/* v=Omega*sigma */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NV,&scale[0],(const float**)&dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]],NV,&beta,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

		if(k<N-1){
			/* v=Theta*q+v */
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NX,&alpha,(const float**)&dev_ptr_Effinet_THETA[TREE_NODES_PER_STAGE_CUMUL[k]],
					NV,(const float**)dev_ptr_q,NX,&alpha,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));
		}
		/* v=Psi*psi+v */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_PSI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],NU,&alpha,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));
		/* v=Phi*xi+v */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_PHI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

		/* r=sigma */
		_CUDA(cudaMemcpy(dev_r,&dev_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV*TREE_NODES_PER_STAGE[k]*sizeof(real_t),cudaMemcpyDeviceToDevice));

		/* r=D*xi+r */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_D[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));

		/* r=f*psi+r */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_F[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],NU,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));

		if(k<N-1){
			/* r=g*q+r */
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NX,&alpha,(const float**)&dev_ptr_Effinet_G[TREE_NODES_PER_STAGE_CUMUL[k]],
					NV,(const float**)dev_ptr_q,NX,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));
		}

		if(k<N-1)
			/* q=F'xi+q */
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,NX,1,2*NX,&alpha,(const float**)&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[k]],
					2*NX,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,dev_ptr_q,NX,TREE_NODES_PER_STAGE[k]));
		else
			/* q=F'xi */
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,NX,1,2*NX,&alpha,(const float**)&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[k]],
					2*NX,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&beta,dev_ptr_q,NX,TREE_NODES_PER_STAGE[k]));

		if(k>0){
			if((TREE_NODES_PER_STAGE[k]-TREE_NODES_PER_STAGE[k-1])>0){
				summation_children<real_t><<<TREE_NODES_PER_STAGE[k-1],NX>>>(dev_q,dev_temp_q,DEV_CONST_TREE_NODES_PER_STAGE,
						DEV_CONSTANT_TREE_NUM_CHILDREN,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,NX,k-1);
				summation_children<real_t><<<TREE_NODES_PER_STAGE[k-1],NV>>>(dev_r,dev_temp_r,DEV_CONST_TREE_NODES_PER_STAGE,
						DEV_CONSTANT_TREE_NUM_CHILDREN,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,NV,k-1);
				_CUDA(cudaMemcpy(dev_r,dev_temp_r,TREE_NODES_PER_STAGE[k-1]*NV*sizeof(real_t),cudaMemcpyDeviceToDevice));
				_CUDA(cudaMemcpy(dev_q,dev_temp_q,TREE_NODES_PER_STAGE[k-1]*NX*sizeof(real_t),cudaMemcpyDeviceToDevice));
			}
		}
	}

	/* Forward substitution */
	_CUDA(cudaMemcpy(dev_u,dev_vhat,N_NODES*NU*sizeof(real_t),cudaMemcpyDeviceToDevice));

	for(int k=0;k<N;k++){
		if(k==0){
			/* x=p, u=h */
			_CUBLAS(cublasSaxpy_v2(handle,NV,&alpha,dev_prev_v,1,dev_v,1));
			_CUDA(cudaMemcpy(dev_x,dev_current_state,NX*sizeof(real_t),cudaMemcpyDeviceToDevice));
			/* x=x+w */
			_CUBLAS(cublasSaxpy_v2(handle,NX,&alpha,dev_disturb_w,1,dev_x,1));
			/* u=Lv+\hat{u} */
			_CUBLAS(cublasSgemv_v2(handle,CUBLAS_OP_N,NU,NV,&alpha,dev_L,NU,dev_v,1,&alpha,dev_u,1));
			/* x=x+Bu */
			_CUBLAS(cublasSgemv_v2(handle,CUBLAS_OP_N,NX,NU,&alpha,dev_B,NX,dev_u,1,&alpha,dev_x,1));

		}else{
			if((TREE_NODES_PER_STAGE[k]-TREE_NODES_PER_STAGE[k-1])>0){
				/* v_k=v_{k-1}+v_k */
				child_nodes_update<real_t><<<TREE_NODES_PER_STAGE[k],NV>>>(&dev_v[TREE_NODES_PER_STAGE_CUMUL[k-1]*NV],&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV]
				                    ,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_ANCESTOR,NV,k-1);
				/* u_k=Lv_k+\hat{u}_k */
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NU,TREE_NODES_PER_STAGE[k],NV,&alpha,dev_L,
						NU,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV,&alpha,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU));
				/* x=w */
				_CUDA(cudaMemcpy(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],&dev_disturb_w[TREE_NODES_PER_STAGE_CUMUL[k]*NX],TREE_NODES_PER_STAGE[k]*NX*
						sizeof(real_t),cudaMemcpyDeviceToDevice));
				/* x=x+Bu */
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NX,TREE_NODES_PER_STAGE[k],NU,&alpha,dev_B,NX,
						&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU,&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],NX));
				/* x_{k+1}=x_k */
				child_nodes_update<real_t><<<TREE_NODES_PER_STAGE[k],NX>>>(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],
						&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_ANCESTOR,NX,k-1);
			}else{
				/* v_k=v_{k-1}+v_k */
				_CUBLAS(cublasSaxpy_v2(handle,NV*TREE_NODES_PER_STAGE[k],&alpha,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k-1]*NV],1,
						&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],1));
				/* u_k=Lv_k+\hat{u}_k */
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NU,TREE_NODES_PER_STAGE[k],NV,&alpha,dev_L,
						NU,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV,&alpha,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU));
				/* x_{k+1}=x_{k} */
				//_CUBLAS(cublasSaxpy_v2(handle,NX*TREE_NODES_PER_STAGE[k],&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],1,
					//	&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1));
				_CUDA(cudaMemcpy(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],NX*TREE_NODES_PER_STAGE[k]*sizeof(real_t),
						cudaMemcpyDeviceToDevice));
				/* x=x+w */
				_CUBLAS(cublasSaxpy_v2(handle,NX*TREE_NODES_PER_STAGE[k],&alpha,&dev_disturb_w[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1,
						&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1));
				/* x=x+Bu */
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NX,TREE_NODES_PER_STAGE[k],NU,&alpha,dev_B,
						NX,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU,&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],NX));
			}
		}
	}



	/*free(ptr_x_c);
	free(x_c);
	free(y_c);*/

}

void test_summation_children(int stage,int dim){

	real_t* host_r=(real_t*)malloc(dim*TREE_NODES_PER_STAGE[stage+1]*sizeof(real_t));
	real_t* host_tem_r=(real_t*)malloc(dim*TREE_NODES_PER_STAGE[stage]*sizeof(real_t));

	for(int i=0;i<dim*TREE_NODES_PER_STAGE[stage+1];i++){
		host_r[i]=1;
	}
	_CUDA(cudaMemcpy(dev_r,host_r,dim*TREE_NODES_PER_STAGE[stage+1]*sizeof(real_t),cudaMemcpyHostToDevice));
	summation_children<real_t><<<TREE_NODES_PER_STAGE[stage],NX>>>(dev_r,dev_temp_r,DEV_CONST_TREE_NODES_PER_STAGE,
			DEV_CONSTANT_TREE_NUM_CHILDREN,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,dim,stage);

	_CUDA(cudaMemcpy(host_tem_r,dev_temp_r,dim*TREE_NODES_PER_STAGE[stage]*sizeof(real_t),cudaMemcpyDeviceToHost));

	for(int i=0;i<dim*TREE_NODES_PER_STAGE[stage];i++){
		assert(host_tem_r[i]-TREE_NODES_PER_STAGE[stage+1]==0);
		//printf("%f ",host_tem_r[i]);
	}
	printf("summation of children is correct \n");
}

void test_multiplication_control(){

	real_t **ptr_x_c=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	real_t *y_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));

	char* filepath_v_temp="Data_files_testing/solve_v_temp.h";

	allocate_data<real_t>(filepath_v_temp,y_c);

	for(int k=N-1;k>-1;k--){

		/* v=Psi*psi */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_PSI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],NU,&beta,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));
		/* v=Phi*xi+v */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_PHI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

	}

	for(int k=0;k<N;k++){
		_CUDA(cudaMemcpy(ptr_x_c,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],TREE_NODES_PER_STAGE[k]*sizeof(real_t*),cudaMemcpyDeviceToHost));
		for(int i=0;i<TREE_NODES_PER_STAGE[k];i++){
			_CUDA(cudaMemcpy(x_c,ptr_x_c[i],NV*sizeof(real_t),cudaMemcpyDeviceToHost));
			for(int j=0;j<NV;j++){
				printf("%f %d ",y_c[(TREE_NODES_PER_STAGE_CUMUL[k]+i)*NV+j]-x_c[j],(TREE_NODES_PER_STAGE_CUMUL[k]+i)*NV+j);
			}
		}
	}
	printf("\n");

	free(ptr_x_c);
	free(x_c);
	free(y_c);

}

void test_multiplication_cost(){
	real_t **ptr_x_c=(real_t**)malloc(N_NODES*sizeof(real_t*));
	real_t *x_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));
	real_t *y_c=(real_t*)malloc(NU*NU*N_NODES*sizeof(real_t));

	char* filepath_r_temp="Data_files_testing/solve_r_temp.h";

	allocate_data<real_t>(filepath_r_temp,y_c);

	for(int k=N-1;k>-1;k--){


		/* v=D*xi */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_D[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],
				2*NX,&beta,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

		/* v=f*psi+v */
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_F[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],
				NU,&alpha,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

	}

	for(int k=0;k<N;k++){
		_CUDA(cudaMemcpy(ptr_x_c,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],TREE_NODES_PER_STAGE[k]*sizeof(real_t*),cudaMemcpyDeviceToHost));
		for(int i=0;i<TREE_NODES_PER_STAGE[k];i++){
			_CUDA(cudaMemcpy(x_c,ptr_x_c[i],NV*sizeof(real_t),cudaMemcpyDeviceToHost));
			for(int j=0;j<NV;j++){
				printf("%f %d ",y_c[(TREE_NODES_PER_STAGE_CUMUL[k]+i)*NV+j]-x_c[j],(TREE_NODES_PER_STAGE_CUMUL[k]+i)*NV+j);
				//printf("%f %d ",x_c[j],(TREE_NODES_PER_STAGE_CUMUL[k]+i)*NU+j);
			}
		}
	}
	printf("\n");

	free(ptr_x_c);
	free(x_c);
	free(y_c);

}

void test_solve_step(){

	real_t tol_solve_step=1;
	real_t* x_test=(real_t*)malloc(NX*sizeof(real_t));
	real_t* W_x=(real_t*)malloc(2*NX*N_NODES*sizeof(real_t));
	real_t* W_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t));
	real_t* Z_x=(real_t*)malloc(NX*N_NODES*sizeof(real_t*));
	real_t* Z_u=(real_t*)malloc(NU*N_NODES*sizeof(real_t*));
	real_t* linear_beta=(real_t*)malloc(NV*N_NODES*sizeof(real_t));
	real_t* vhat_test=(real_t*)malloc(NU*N_NODES*sizeof(real_t));
	real_t* disturb_w_test=(real_t*)malloc(NX*N_NODES*sizeof(real_t*));
	real_t* prev_v_test=(real_t*)malloc(NV*sizeof(real_t*));

	char* filepath_x="Data_files_testing/solve_x.h";
	char* filepath_W_x="Data_files_testing/solve_W_x.h";
	char* filepath_W_u="Data_files_testing/solve_W_u.h";
	char* filepath_Z_x="Data_files_testing/solve_Z_x.h";
	char* filepath_Z_u="Data_files_testing/solve_Z_u.h";
	char* filepath_beta="Data_files_testing/solve_beta.h";
	char* filepath_vhat="Data_files_testing/solve_vhat.h";
	char* filepath_w="Data_files_testing/solve_w.h";
	char* filepath_v="Data_files_testing/solve_v.h";

	allocate_data<real_t>(filepath_x,x_test);
	allocate_data<real_t>(filepath_W_x,W_x);
	allocate_data<real_t>(filepath_W_u,W_u);
	allocate_data<real_t>(filepath_Z_x,Z_x);
	allocate_data<real_t>(filepath_Z_u,Z_u);
	allocate_data<real_t>(filepath_beta,linear_beta);
	allocate_data<real_t>(filepath_vhat,vhat_test);
	allocate_data<real_t>(filepath_w,disturb_w_test);
	allocate_data<real_t>(filepath_v,prev_v_test);

	_CUDA(cudaMemcpy(dev_current_state,x_test,NX*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_prev_v,prev_v_test,NV*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_linear_cost_b,linear_beta,NV*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_disturb_w,disturb_w_test,NX*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_vhat,vhat_test,NU*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_accelarated_xi,W_x,2*NX*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_accelarated_psi,W_u,NU*N_NODES*sizeof(real_t),cudaMemcpyHostToDevice));

	//check_correctness_memcpy<real_t>(disturb_w_test,dev_disturb_w,NX*N_NODES,1);
	//check_correctness_memcpy<real_t>(W_x,dev_accelarated_xi,2*NX*N_NODES,1,tol_solve_step);
	//check_correctness_memcpy<real_t>(W_u,dev_accelarated_psi,NU*N_NODES,1,tol_solve_step);

	//test_multiplication_control();
	//test_multiplication_cost();
	effinet_solve_step();

	check_correctness_memcpy<real_t>(Z_u,dev_u,NU*N_NODES,1,tol_solve_step);
	//tol_solve_step=1e-2;
	check_correctness_memcpy<real_t>(Z_x,dev_x,NX*N_NODES,1,tol_solve_step);
	/*
	for(int k=0;k<N_NODES*NX;k++){
		printf("%f %d ",Z_x[k],k);
	}
	printf("\n");
	_CUDA(cudaMemcpy(Z_x,dev_x,NX*N_NODES*sizeof(real_t),cudaMemcpyDeviceToHost));
	for(int k=0;k<N_NODES*NX;k++){
		printf("%f %d ",Z_x[k],k);
		//printf("%f %d ",W_x[k],k);
	}
	printf("\n");*/

	free(x_test);
	free(W_x);
	free(W_u);
	free(Z_x);
	free(Z_u);
	free(linear_beta);
	free(disturb_w_test);
	free(prev_v_test);
	free(vhat_test);
	printf("testing of solve step is done \n");

}


#endif /* EFFINET_SOLVE_STEP_CUH_ */
