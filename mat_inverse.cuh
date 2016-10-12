/*
 * mat_inverse.cuh
 *
 *  Created on: Jul 24, 2015
 *      Author: ajay
 */

#ifndef MAT_INVERSE_CUH_
#define MAT_INVERSE_CUH_

#include "api_effinet_cuda.cuh"

void invert(real_t** src, real_t** dst, uint_t n, uint_t batchSize)
{
	cublasHandle_t handle;
	_CUBLAS(cublasCreate(&handle));

	uint_t *P, *INFO;

	_CUDA(cudaMalloc(&P, n * batchSize * sizeof(uint_t)));
	_CUDA(cudaMalloc(&INFO, batchSize * sizeof(uint_t)));

	uint_t lda = n;

	real_t **A = (real_t **)malloc(batchSize*sizeof(real_t *));
	real_t **A_d, *A_dflat;
	_CUDA(cudaMalloc(&A_d,batchSize*sizeof(real_t *)));
	_CUDA(cudaMalloc(&A_dflat, n*n*batchSize*sizeof(real_t)));
	A[0] = A_dflat;
	for (uint_t i = 1; i < batchSize; i++)
		A[i] = A[i-1]+(n*n);
	_CUDA(cudaMemcpy(A_d,A,batchSize*sizeof(real_t *),cudaMemcpyHostToDevice));
	for (uint_t i = 0; i < batchSize; i++)
		_CUDA(cudaMemcpy(A_dflat+(i*n*n), src[i], n*n*sizeof(real_t), cudaMemcpyHostToDevice));

	_CUBLAS(cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

	uint_t INFOh[batchSize];
	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));

	for (uint_t i = 0; i < batchSize; i++)
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	real_t **C = (real_t **)malloc(batchSize*sizeof(real_t *));
	real_t **C_d, *C_dflat;
	_CUDA(cudaMalloc(&C_d,batchSize*sizeof(real_t *)));
	_CUDA(cudaMalloc(&C_dflat, n*n*batchSize*sizeof(real_t)));
	C[0] = C_dflat;
	for (uint_t i = 1; i < batchSize; i++)
		C[i] = C[i-1] + (n*n);
	_CUDA(cudaMemcpy(C_d,C,batchSize*sizeof(real_t *),cudaMemcpyHostToDevice));
	_CUBLAS(cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,lda,INFO,batchSize));

	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));

	for (uint_t i = 0; i < batchSize; i++)
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	for (uint_t i = 0; i < batchSize; i++)
		_CUDA(cudaMemcpy(dst[i], C_dflat + (i*n*n), n*n*sizeof(real_t), cudaMemcpyDeviceToHost));
	cudaFree(A_d); cudaFree(A_dflat); free(A);
	cudaFree(C_d); cudaFree(C_dflat); free(C);
	cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}

void inverse_batch(real_t** src,real_t** dst,uint_t n,uint_t batchSize,cublasHandle_t handle){
	uint_t *P, *INFO;

	_CUDA(cudaMalloc((void**)&P, n * batchSize * sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&INFO, batchSize * sizeof(uint_t)));

	uint_t lda = n;

	real_t** x=(real_t**)malloc(batchSize*sizeof(real_t*));
	real_t* y=(real_t*)malloc(n*n*sizeof(real_t));


	_CUBLAS(cublasSgetrfBatched(handle,n,src,lda,P,INFO,batchSize));

	/*
	_CUDA(cudaMemcpy(x,src,batchSize*sizeof(real_t*),cudaMemcpyDeviceToHost));
	for(int kk=0;kk<batchSize;kk++){
		_CUDA(cudaMemcpy(y,x[kk],n*n*sizeof(real_t),cudaMemcpyDeviceToHost));
		for (int j=0;j<n*n;j++){
			printf("%f ",y[j]);
		}
		printf("\n");
	}*/

	uint_t INFOh[batchSize];

	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));
	for (uint_t i = 0; i < batchSize; i++){
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

	_CUBLAS(cublasSgetriBatched(handle,n,(const float **)src,lda,P,dst,lda,INFO,batchSize));
	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));

	for (uint_t i = 0; i < batchSize; i++)
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	_CUDA(cudaFree(P));
	_CUDA(cudaFree(INFO));
}
void test_inverse(){
	uint_t size_n= 3;
	uint_t batch_size=1;

	real_t* matA=(real_t*)malloc(size_n*size_n*batch_size*sizeof(real_t));
	real_t* inv_matA=(real_t*)malloc(size_n*size_n*batch_size*sizeof(real_t));
	real_t** ptr_matA=(real_t**)malloc(batch_size*sizeof(real_t*));
	real_t** ptr_inv_matA=(real_t**)malloc(batch_size*sizeof(real_t*));

	real_t *dev_matA,*dev_inv_matA,**dev_ptr_matA,**dev_ptr_inv_matA;

	_CUDA(cudaMalloc((void**)&dev_matA,batch_size*size_n*size_n*sizeof(real_t)));
	_CUDA(cudaMalloc((void**)&dev_inv_matA,batch_size*size_n*size_n*sizeof(real_t)));

	_CUDA(cudaMalloc((void**)&dev_ptr_matA,batch_size*sizeof(real_t*)));
	_CUDA(cudaMalloc((void**)&dev_ptr_inv_matA,batch_size*sizeof(real_t*)));

	real_t temp;
	for(int k=0;k<batch_size;k++){
		for(int i=0;i<size_n;i++)
			for(int j=0;j<size_n;j++){
				temp=(real_t)(rand() % 29)/32;
				matA[k*size_n*size_n+i*size_n+j]=temp;
				if(i==j){
					matA[k*size_n*size_n+i*size_n+j]=0.5;
				}
			}
		ptr_matA[k]=&dev_matA[k*size_n*size_n];
		ptr_inv_matA[k]=&dev_inv_matA[k*size_n*size_n];
	}
	for(int k=0;k<batch_size;k++){
		printf("matrix :%d \n",k);
		for(int i=0;i<size_n;i++){
			for(int j=0;j<size_n;j++){
				printf("%f ",matA[k*size_n*size_n+i*size_n+j]);
			}
			printf("\n");
		}
	}

	_CUDA(cudaMemcpy(dev_matA,matA,batch_size*size_n*size_n*sizeof(real_t),cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(dev_ptr_matA,ptr_matA,batch_size*sizeof(real_t*),cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(dev_ptr_inv_matA,ptr_inv_matA,batch_size*sizeof(real_t*),cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	_CUBLAS(cublasCreate(&handle));

	inverse_batch(dev_ptr_matA,dev_ptr_inv_matA,size_n,batch_size,handle);
	_CUDA(cudaMemcpy(inv_matA,dev_inv_matA,batch_size*size_n*size_n*sizeof(real_t),cudaMemcpyDeviceToHost));
	for(int k=0;k<batch_size;k++){
		printf("inverse of matrix :%d \n",k);
		for(int i=0;i<size_n;i++){
			for(int j=0;j<size_n;j++){
				printf("%f ",inv_matA[k*size_n*size_n+i*size_n+j]);
			}
			printf("\n");
		}
	}
	printf("Test successful\n");

	free(matA);
	free(ptr_matA);
	free(inv_matA);
	free(ptr_inv_matA);

	_CUDA(cudaFree(dev_matA));
	_CUDA(cudaFree(dev_inv_matA));
	_CUDA(cudaFree(dev_ptr_matA));
	_CUDA(cudaFree(dev_ptr_inv_matA));

}



#endif /* MAT_INVERSE_CUH_ */
