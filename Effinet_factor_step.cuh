/*
 * Effinet_factor_step.cuh
 * The factor step and related functions are implemented in this file.
 *  Created on: Jul 28, 2015
 *      Author: ajay
 *      
 * This file has the implementation of the factor step -- Appendix B  
 * List of funcitons:
 * 1) void  effinet_factor_step(void);
 * 2) void test_factor_step(void);
 * 3) void check_correctness_memcpy(T* x,T *dev_x,int size,int alpha,real_t tol);
 */
#ifndef EFFINET_FACTOR_STEP_CUH_
#define EFFINET_FACTOR_STEP_CUH_
#include "mat_inverse.cuh"
#include "Effinet_data_generation.cuh"

void  effinet_factor_step(void){
	real_t scale[2]={-0.5,1};
	real_t *dev_Bbar,*dev_Gbar;
	real_t **dev_ptr_Bbar,**dev_ptr_Gbar,**ptr_Bbar,**ptr_Gbar;

	real_t** x=(real_t**)malloc(K*sizeof(real_t*));
	real_t* y=(real_t*)malloc(2*NU*NU*sizeof(real_t));

	_CUDA(cudaMalloc((void**)&dev_Bbar,NV*NX*K*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_Gbar,NU*NV*K*sizeof(real_t)));

	ptr_Bbar=(real_t**)malloc(K*sizeof(real_t*));
	ptr_Gbar=(real_t**)malloc(K*sizeof(real_t*));

	_CUDA(cudaMalloc((void**)&dev_ptr_Bbar,K*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_Gbar,K*sizeof(real_t*)));

	for(int i=0;i<K;i++){
		ptr_Bbar[i]=&dev_Bbar[i*NX*NV];
		ptr_Gbar[i]=&dev_Gbar[i*NX*NV];
	}

	_CUDA(cudaMemcpy(dev_ptr_Gbar,ptr_Gbar,K*sizeof(real_t*),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_Bbar,ptr_Bbar,K*sizeof(real_t*),cudaMemcpyHostToDevice));

	/*
	inverse_batch(&dev_ptr_R[TREE_NODES_PER_STAGE_CUMUL[N-1]],&dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[N-1]],NV,TREE_NODES_PER_STAGE[N-1],handle);

	_CUDA(cudaMemcpy(x,&dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[N-1]],TREE_NODES_PER_STAGE[N-1]*sizeof(real_t*),cudaMemcpyDeviceToHost));
	for(int kk=0;kk<TREE_NODES_PER_STAGE[N-1];kk++){
		_CUDA(cudaMemcpy(y,x[kk],NV*NV*sizeof(real_t),cudaMemcpyDeviceToHost));
		for (int j=0;j<NV*NV;j++){
			printf("%f ",y[j]);
		}
		printf("\n");
	}

		_CUDA(cudaMemcpy(x,&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[i]],TREE_NODES_PER_STAGE[i]*sizeof(real_t*),cudaMemcpyDeviceToHost));
		for(int kk=0;kk<TREE_NODES_PER_STAGE[i];kk++){
			_CUDA(cudaMemcpy(y,x[kk],2*NX*NX*sizeof(real_t),cudaMemcpyDeviceToHost));
			for (int j=0;j<2*NX*NX;j++){
				printf("%f ",y[j]);
			}
			printf("\n");
		}
	/* Bbar'*/
	_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_T,NV,NX,NU,&alpha,(const float**)dev_ptr_L,NU,
						(const float**)dev_ptr_B,NX,&beta,dev_ptr_Bbar,NV,K));

	for(int i=N-1;i>-1;i--){

		/* omega=(p_k\bar{R})^{-1}*/
		inverse_batch(&dev_ptr_R[TREE_NODES_PER_STAGE_CUMUL[i]],&dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i],handle);

		/* effinet_f=GBar*/
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_T,NV,NU,NU,&alpha,(const float**)dev_ptr_L,NU,
							(const float**)&dev_ptr_G[TREE_NODES_PER_STAGE_CUMUL[i]],NU,&beta,
							&dev_ptr_Effinet_F[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i]));

		/* effinet_g=\bar{B}'*/
		_CUDA(cudaMemcpy(&dev_Effinet_G[NX*NV*TREE_NODES_PER_STAGE_CUMUL[i]],dev_Bbar,NX*NV*TREE_NODES_PER_STAGE[i]*sizeof(real_t),cudaMemcpyDeviceToDevice));

		/* effinet_d=\bar{B}'F'*/
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,NV,2*NX,NX,&alpha,(const float**)dev_ptr_Bbar,NV,
							(const float**)&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[i]],2*NX,&beta,
							&dev_ptr_Effinet_D[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i]));

		/* phi=\omega \bar{B}'F'*/
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,2*NX,NV,&scale[0],(const float**)&
				dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[i]],NV,(const float**)&dev_ptr_Effinet_D[TREE_NODES_PER_STAGE_CUMUL[i]],NV,&beta,
							&dev_ptr_Effinet_PHI[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i]));

		/* theta=\omega \bar{B}'*/
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,NX,NV,&scale[0],(const float**)&
				dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[i]],NV,(const float**)dev_ptr_Bbar,NV,&beta,
							&dev_ptr_Effinet_THETA[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i]));

		/* psi=\omega \bar{G}'*/
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,NU,NV,&scale[0],(const float**)&
				dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[i]],NV,(const float**)&dev_ptr_Effinet_F[TREE_NODES_PER_STAGE_CUMUL[i]],NV,&beta,
							&dev_ptr_Effinet_PSI[TREE_NODES_PER_STAGE_CUMUL[i]],NV,TREE_NODES_PER_STAGE[i]));
	}
	printf("Effinet factor step is completed\n");
	free(ptr_Bbar);
	free(ptr_Gbar);

	_CUDA(cudaFree(dev_Bbar));
	_CUDA(cudaFree(dev_Gbar));
	_CUDA(cudaFree(dev_ptr_Bbar));
	_CUDA(cudaFree(dev_ptr_Gbar));

}

void test_factor_step(void){

	real_t tol_factor_step=1e-3;
	real_t* PHI=(real_t*)malloc(2*NX*NV*N_NODES*sizeof(real_t));
	real_t* PSI=(real_t*)malloc(NV*NU*N_NODES*sizeof(real_t));
	real_t* THETA=(real_t*)malloc(NV*NX*N_NODES*sizeof(real_t*));
	real_t* OMEGA=(real_t*)malloc(NV*NV*N_NODES*sizeof(real_t*));
	real_t* D=(real_t*)malloc(2*NV*NX*N_NODES*sizeof(real_t*));
	real_t* F=(real_t*)malloc(NV*NU*N_NODES*sizeof(real_t*));
	real_t* G=(real_t*)malloc(NV*NX*N_NODES*sizeof(real_t*));

	char* filepath_omega="Data_files_testing/Ptree_omega.h";
	char* filepath_theta="Data_files_testing/Ptree_theta.h";
	char* filepath_phi="Data_files_testing/Ptree_phi.h";
	char* filepath_psi="Data_files_testing/Ptree_psi.h";
	char* filepath_D="Data_files_testing/Ptree_d.h";
	char* filepath_F="Data_files_testing/Ptree_f.h";
	char* filepath_G="Data_files_testing/Ptree_g.h";

	allocate_data<real_t>(filepath_phi,PHI);
	allocate_data<real_t>(filepath_psi,PSI);
	allocate_data<real_t>(filepath_theta,THETA);
	allocate_data<real_t>(filepath_omega,OMEGA);
	allocate_data<real_t>(filepath_D,D);
	allocate_data<real_t>(filepath_F,F);
	allocate_data<real_t>(filepath_G,G);

	check_correctness_memcpy<real_t>(OMEGA,dev_Effinet_OMEGA,NV*NV*N_NODES,-2,tol_factor_step);
	check_correctness_memcpy<real_t>(PHI,dev_Effinet_PHI,2*NX*NV*N_NODES,1,tol_factor_step);
	check_correctness_memcpy<real_t>(PSI,dev_Effinet_PSI,NU*NV*N_NODES,1,tol_factor_step);
	check_correctness_memcpy<real_t>(THETA,dev_Effinet_THETA,NX*NV*N_NODES,1,tol_factor_step);
	check_correctness_memcpy<real_t>(D,dev_Effinet_D,2*NX*NV*N_NODES,1,tol_factor_step);
	check_correctness_memcpy<real_t>(F,dev_Effinet_F,NU*NV*N_NODES,1,tol_factor_step);
	check_correctness_memcpy<real_t>(G,dev_Effinet_G,NX*NV*N_NODES,1,tol_factor_step);

	free(PHI);
	free(PSI);
	free(THETA);
	free(OMEGA);
	free(D);
	free(F);
	free(G);

}

template<typename T>void check_correctness_memcpy(T* x,T *dev_x,int size,int alpha,real_t tol){

	T *y=(T*)malloc(size*sizeof(T));
	_CUDA(cudaMemcpy(y,dev_x,size*sizeof(T),cudaMemcpyDeviceToHost));
	for(int i=0;i<size;i++){
		if(fabs(alpha*x[i]-y[i])>tol){
			//printf("%f %d ",fabs(alpha*x[i]-y[i]),i);
			printf("%f %d ",alpha*x[i]-y[i],i);
		}
	}
	free(y);
	printf("SUCESS \n");
}

#endif /* EFFINET_FACTOR_STEP_CUH_ */
