/*
 * api_effinet_cuda.cuh
 *  Created on: Jul 24, 2015
 *      Author: ajay
 *  
 *  This file is the API of the data and functions used in the APG algorithm 
 *  for the Effinet project.
 */


#ifndef API_EFFINET_CUDA_CUH_
#define API_EFFINET_CUDA_CUH_


#include "effinet_header.h"

#define _CUDA(call) \
		do \
		{ \
			cudaError_t err = (call); \
			if(cudaSuccess != err) \
			{ \
				fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
				cudaDeviceReset(); \
				exit(EXIT_FAILURE); \
			} \
		} \
		while (0)

#define _CUBLAS(call) \
		do \
		{ \
			cublasStatus_t status = (call); \
			if(CUBLAS_STATUS_SUCCESS != status) \
			{ \
				fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status); \
				cudaDeviceReset(); \
				exit(EXIT_FAILURE); \
			} \
			\
		} \
		while(0)


/**** Constant device variables ****/
/**
 * Stages of the tree (constant device data).
 */
uint_t* DEV_CONST_TREE_STAGES;
/**
 * Nodes per stage (constant device data).
 */
uint_t* DEV_CONST_TREE_NODES_PER_STAGE;
/**
 * Leave nodes of the tree (constant device data).
 */
uint_t* DEV_CONST_TREE_LEAVES;
/**
 * Children of each node (constant device data).
 */
uint_t* DEV_CONST_TREE_CHILDREN;
/*
 * Cumulative nodes until the stage
 */
uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL;
/*
 * Number of children for each node (except the leave node)
 */
uint_t* DEV_CONSTANT_TREE_NUM_CHILDREN;
/*
 * Cumulative children until the stage.
 */
uint_t* DEV_CONSTANT_TREE_N_CHILDREN_CUMUL;
/*
 * Ancestor of each node in the tree.
 */
uint_t* DEV_CONSTANT_TREE_ANCESTOR;
/*
 * The value at each node of the tree
 */
real_t*  dev_TREE_VALUE;
real_t** dev_ptr_Tree_Value;

/** System data*/
real_t* dev_A;
real_t* dev_B;
real_t* dev_F;
real_t* dev_G;
real_t* dev_xmin;
real_t* dev_xmax;
real_t* dev_xs;
real_t* dev_umin;
real_t* dev_umax;
real_t* dev_R;
real_t* dev_L;

/** pointers to system*/
real_t** dev_ptr_A;
real_t** dev_ptr_B;
real_t** dev_ptr_F;
real_t** dev_ptr_G;
real_t** dev_ptr_xmin;
real_t** dev_ptr_xmax;
real_t** dev_ptr_xs;
real_t** dev_ptr_umin;
real_t** dev_ptr_umax;
real_t** dev_ptr_R;
real_t** dev_ptr_L;

/** Data corresponding to the present state of the system*/
real_t* dev_current_state;/* */
real_t* dev_prev_vhat;
real_t* dev_prev_v;
real_t* dev_vhat;
real_t* dev_linear_cost_b;
real_t* dev_disturb_w;

/** pointers to present state of the system*/
real_t** dev_ptr_prev_vhat;
real_t** dev_ptr_prev_v;
real_t** dev_ptr_vhat;
real_t** dev_ptr_linear_cost_b;
real_t** dev_ptr_disturb_w;


/** device variables for solving */
real_t* dev_x;
real_t* dev_u;
real_t* dev_v;

real_t* dev_q;
real_t* dev_r;
real_t* dev_temp_q;
real_t* dev_temp_r;

real_t* dev_xi;
real_t* dev_psi;
real_t* dev_update_xi;
real_t* dev_update_psi;
real_t* dev_accelarated_xi;
real_t* dev_accelarated_psi;

real_t* dev_primal_xi;
real_t* dev_primal_psi;
real_t* dev_dual_xi;
real_t* dev_dual_psi;

/** Array of pointers of primal and dual parameters at each node */
real_t** dev_ptr_x;
real_t** dev_ptr_u;
real_t** dev_ptr_v;

real_t** dev_ptr_q;
real_t** dev_ptr_r;

real_t** dev_ptr_accelerated_xi;
real_t** dev_ptr_accelerated_psi;

real_t** dev_ptr_primal_xi;
real_t** dev_ptr_primal_psi;

/** Algorithm details */

/** parameters used in cublas library*/
real_t alpha=1.0;
real_t beta=0.0;
real_t al_n=-1.0;
/** used to find the minimum in cuda library*/
int index_min;

/** constants used in the algorithm */
real_t dual_gap=0.005;
real_t primal_inf=0.005;
uint_t GPAD_iterates=0;
uint_t GPAD_first_cond_flag=0;
uint_t GPAD_second_cond_flag=0;
real_t GPAD_primal_inf_term;
real_t lambda;
real_t inv_lambda;

/** Default options */

/** theta_restart option allows you to select which mode to operate
    @param         0   enforce just monotonicity (default)
    @param         1   restart theta=1 when the condition is satisfied */
uint_t theta_restart=0;

/**  restart_method option allows you to select the monotonic condition
      @param       0   Gradient based test (default)
      @param       1   Dual cost based test  */
uint_t restart_method=0;


/** solver parameters */
real_t *dev_primal; /** parameter used to calculate the primal infesibility average*/
real_t *dev_primal_avg; /** primal average */
real_t *dev_primal_iterate; /** primal infeasibility of the iterate*/
real_t ntheta[2]={1,1}; /** theta, the accelerating variable*/

real_t *grad;    /** Gradient at each stage */
real_t prm_cst[2]; /** Primal cost at each stage*/
real_t dual_cst[2]; /** Dual cost at each stage*/
real_t *dev_temp_cst; /** Temporary variable used in calculation of primal cost */

real_t *prm_cst_value;/** variable to store the primal cost during the algorithm (optional)*/
real_t *dual_cst_value; /** variable to store the dual cost during the algorithm (optional)*/
real_t *prm_inf; /** variable to store the primal infeasibility during the algorithm (optional)*/

/*
real_t *dev_temp_xcst;
real_t *dev_temp_ucst;
real_t **dev_ptr_temp_xcst; /** pointer of temporary cost of the primal cost*/
/* real_t **dev_ptr_temp_ucst; /** pointer of temporary cost of the primal cost*/


/** solve-step data*/
real_t* dev_Effinet_PHI;
real_t* dev_Effinet_THETA;
real_t* dev_Effinet_PSI;
real_t* dev_Effinet_SIGMA;
real_t* dev_Effinet_OMEGA;
real_t* dev_Effinet_D;
real_t* dev_Effinet_F;
real_t* dev_Effinet_G;

/** solve-step data pointers*/
real_t** dev_ptr_Effinet_PHI;
real_t** dev_ptr_Effinet_THETA;
real_t** dev_ptr_Effinet_PSI;
real_t** dev_ptr_Effinet_SIGMA;
real_t** dev_ptr_Effinet_OMEGA;
real_t** dev_ptr_Effinet_D;
real_t** dev_ptr_Effinet_F;
real_t** dev_ptr_Effinet_G;

/**cublas data */

cublasHandle_t handle; /* Handle to perform the cublas operations */

/** Functions */


/**
 * Function   :    accelerated_dual_update
 * Details : calculates the accelerated dual vector
 *           w_{v}=y_{v}+\theta_{v}(\theta_{v-1}^{-1}-1)(y_{v}-y_{v-1})
 *           and update y_{v-1} with y_{v}-- equation 26(a) 
 *
 * @param     y1                 y_{v}
 * @param     y2                 y_{v-1}
 * @param     alpha              \theta_{v}*(\theta_{v-1}^{-1}-1)
 * @param     w                  w_v
 */
template<typename T>__global__ void accelerated_dual_update(T *dual_k,T *dual_k_1,T *accelerated_dual,T alpha,int size);


/**
 * Function   : summation_children
 * Details : performs the summation of the x with all the siblings and 
 *           stores in the parent node. 
 *
 * @param          dev_vecX        input vector
 * @param          dev_vecY        output vector
 * @param          dim             dimension of the vector each of the children
 * @param          stage           stage of the tree.
 */

template<typename T> __global__ void summation_children(T *dev_vecX,T *dev_vecY,
		uint_t* DEV_CONST_TREE_NODES_PER_STAGE,uint_t* DEV_CONSTANT_TREE_NUM_CHILDREN,
		uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,uint_t* DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,uint_t dim,uint_t stage);

/**
 *  Function    :  child_nodes_update
 *  Details : calculates the state at all the children nodes by
 *            adding the respective values from TREE_VALUE.
 *  @param            x           input vector
 *  @param            y           output vector
 *  @param        stage           stage of the tree
 *
 */
template<typename T>__global__ void child_nodes_update(T *x,T *y,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,
		uint_t* DEV_CONSTANT_TREE_ANCESTOR,int dim,int stage);

/**
 *  Function  :   copy_vector_children 
 *  Details : copies the parent node values to the children nodes at a stage K
 *
 *  @param           x        input vector
 *  @param           y        output vector
 *  @param         dim        dimension of the vector
 *  @param       stage        stage of the tree
 *
 */
template<typename T>__global__ void copy_vector_childeren(T *x,T *y,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,
		uint_t* DEV_CONSTANT_TREE_ANCESTOR,uint dim,int stage);

/**
 * Funciton   :   effinet_proximal_function_g();
 * Details: Implements proximal function with respect to g 
 * Equation 26(c)
 */

__host__ void effinet_proximal_function_g();


/**
 *  Function   :   projection_state
 *  Details: Implement the proximal function for the state  (here it is a projection)
 *  y_{v+1}=[w_{v}+\alpha*(x-g)
 *  @param        x       dual vector
 *  @param        y       gN in the constraints
 *  @param        w       accelerated dual vector
 *  @param   primal       calculates g(z)
 *  @param    alpha       Lipchits constant
 */
template<typename T>__global__ void projection_state(T *x,T *lb,T *ub,T *safety_level,int size);

/**
 *  Function   :  projection_state
 *  Details: Implement the proximal function for the control (here it is a projection) 
 *  y_{v+1}=[w_{v}+\alpha*(x-g)
 *  @param        x       dual vector
 *  @param        y       gN in the constraints
 *  @param        w       accelerated dual vector
 *  @param   primal       calculates g(z)
 *  @param    alpha       Lipchits constant
 */
template<typename T>__global__ void projection_control(T *u,T *lb,T *ub,int size);

/**
 * Function   :   dual_update
 * Details: update the dual variable -- Equation 26(d)
 * 
 * @param      accelerate_dual        accelerated dual vector
 * @param      primal_dual (Hx)       smooth function primal feasibility
 * @param      z_dual (z)             non-smooth vector
 * @param      step_size              step size
 * @param      size                   size of the vector
 */
template<typename T>__global__ void dual_update(T *accelerated_dual,T *primal_dual,T *z_dual,T *update_dual,
		float step_size,int size);
/**
 *  Function   :   find max
 *  Details : max element in the vector is found and store in vector y. 
 *            The first element of y is the maximum element.
 *  @param         x       Actual vector
 *  @param         y       Vector in Descending order
 *  @param    length       The length of the vector
 *  @param    nthreads     half of the length
 */
template<typename T>__global__ void find_max(T *x,T *y,int lenght,int nthreads);

/**
 * Function      :     allocate_data
 * Details : transfers the data from the file to the variable
 * @param     filepath       char containing the file path
 * @param     data           variable to which this data need to be stored (host).
 */
template<typename T>int allocate_data(char*  filepath,T* data);

/**
 *  Function    :    transfer_data_gpu
 *  Details : transfers the data from the file to variable in GPU
 *  @param       dev_x          pointer containing the address of the variable in GPU
 *  @param       x              pointer containing the address of the variable in CPU
 *  @param       filepath       address of the file containing the data
 *  @param       size           variables in the file
 *  @param       repeted_times  number of times the variable needed to be allocated in GPU
 */
template<typename T>void transfer_data_gpu(T *dev_x,T *x,char* filepath,int size,int repeted_times);

/**
 * Function   :  create_tree_gpu
 * Details : creates a tree structure in gpu
 * 
 */
void create_tree_gpu(void);

/**
 * Function     :   create_effinet_gpu()
 * Details : create the effinet problem in GPU
 * @param      handle       cublas instance of the library
 */
//void create_effinet_gpu(void);
void create_effinet_gpu(cublasHandle_t handle);

/**
 *  Function    :    calcualte_particular_solution
 *  Details :  allocates the particular solution in gpu.
 */
template<typename T>void calculate_particular_solution();

/**
 * Function   :    allocate_factor_step_gpu();
 * Details : allocate factor step matrices in gpu
 */

void allocate_factor_step_gpu(void);

/**
 * Function   :    allocate_solve_step_gpu
 * Details : allocate the solve step variables in gpu
 */

void allocate_solve_step_gpu(void);

/**
 * Function : effinet_factor_step
 * Details :  Calculates the off-line matrices required for the solve step; 
 *            Appendix 2 in the paper.
 */
void effinet_factor_step(void);

/**
 * Function : effinet_solve_step(void)
 * Details : Implement the solve step that calculates perform the
 *           on-line computations for dual gradient computation. 
 *           This is the Algorithm 1 in the paper
 */
void effinet_solve_step(void);

/**
 * Function : APG_algorithm
 * Details: Implementation of acceleration gradient algorithm 
 *          on the effinet sytem -- equation 26.
 * @param    current_x    current state of the network
 * @param    pre_v        previous control v
 * @param    handle       handle to perform cublas library
 */
__host__ void APG_algorithm();

/**
 * Function : check_correctness_memcpy
 * Details : check the value at x against dev_x: max(alpha*x-dev_x)<tolerance
 * 
 * @param    x            Host pointer
 * @param    dev_x        GPU memory pointer
 * @param    size         size of the memory in GPU.
 * @param    alpha        alpha
 * @param    tol          tolerance
 */

template<typename T>void check_correctness_memcpy(T* x,T *dev_x,int size,int alpha,real_t tol);

/**
 * Function : test_factor_step
 * Details : tests the factor step of the APG algorithm
 */
void test_factor_step(void);

/**
 * Function : test_solve_step
 * Details : tests the solve step of the APG algorithm
 */
void test_solve_step(void);

/**
 * Function : test_summation_children
 * Details : Test the summation_children kernal used in the solve step;
 *           Check Algorithm 1
 * @param     stage    stage of the tree
 * @param     dim      dimension of the vector
 */

void test_summation_children(int stage,int dim);

/* *
 * Function :  invert 
 * Details  : calculates the inverse of the batch of matrices
 *            using LU-decompostion and stores the result in the dst
 *
 * @param       ptr_matA         pointer to the matrices(host pointers )
 * @param       ptr_inv_matA     pointer to the inverse of the matrices(host pointers)
 * @param       n                dimension of the matrices
 * @param       batchsize        batchsize denotes the size of the number of the matrices
 *
 */
void invert(real_t** ptr_matA, real_t** ptr_inv_matA, uint_t n, uint_t batchSize);

/**
 *  Function  : system_update 
 *  Details  : updates the system in forward substitution
 *             Check the Algorithm 1 in the paper.
 *  @param       handle       cublas handle
 *  @param       stage        stage of the tree
 */

template<typename T>__host__ void system_update(cublasHandle_t handle,int stage);

/**
 * Function : calculate_cost
 * Details  : calculate the primal and dual cost with the primal and dual
 *            solution
 * @param                handle          cublas handle
 */

template<typename T>__host__ void calculate_cost(cublasHandle_t handle);


/**
 * Function    :   free_tree_data
 * Details :  free the TREE STRUCTURE from the gpu.
 */
void free_tree_gpu(void);

/**
 * Function    :    free_effinet_gpu
 * Details : free the effinet system from the gpu. This includes
 *           the particular solution varibables also.
 */
void free_effinet_gpu(void);

/**
 * Function    :     free_factor_step
 * Details : Free the factor step marices from the gpu.
 */
void free_factor_step(void);

/**
 *  Function    :    free_solve_step
 *  Details : Free the solve step matrices from the gpu.
 */
void free_solve_step(void);

/**
 * Funciton    :     free_host_mem()   
 * Details : Free the system memory in the host.
 */
void free_host_mem();


#endif /* API_EFFINET_CUDA_CUH_ */
