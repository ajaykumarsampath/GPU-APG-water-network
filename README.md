# GPU-APG-Water-network 
The objective of this project is to incorporate the stochastic nature of 
the water demand for economical managment the water networks. A stochastic model predictive controller
considers the probabilistic nature of the water demand in deciding the control
policy. But the main hinderance for practical implementation of the stochastic
predictive control in water network is lack of proper optimisation algorithms
that can solve this problem in real-time. To overcome this limitation we 
proposed an optimisation algotithm that exploits the structure of a stochastic
optimal control problem and suitable for parallelisation. This method is 
based on accelerated proximal gradient (APG) algorithm and uses the dual
formulation to solve the problem. Detailed explaination about theory and 
algorithm is discussed in the paper https://arxiv.org/abs/1604.01074.


This repository containts the CUDA-C implementation of the accelerated proximal
gradient (APG) algorithm to solve the stochastic predicitve control problem on
the water networks of Barcelona. This work is done as a part of the EFFINET project. 
The routines defined are based on the Algorithms in the paper.


# Explanation details 
The file main_effinet.cu contains the main function. The list of 
functions is given in the file api_effinet_cuda.cuh. Before executing the 
actual algorithm, memory for all the variables involved is allocated 
in GPU. This includes the system state, control, system dynamic matrices,
tree strucute, cost function and constraints on state and control input. 
All the computations that re involved in the APG algorithm are matrice-vecotor 
multiplications. The matrices that are invovled depends only on the 
system dynamics, constraints, cost function and tree structure. Therefore 
these matrices need not be calculated every sampling time. Appendix B of the paper
provide details about these matrices. These matrices are implemented in the file 
Effinet_factor_step.cuh. Note that these calculation need to be preformed before
the actual APG algorithm and need not be accounted in the actual runtime
of the algorithm. 

The steps of the algorithm APG algorithm is given by Equation 26 of the
paper. This is implemented in the file Effinet_APG.cuh. For each step in 
the algorithm, we defined a function -- for the extrapolation step Equation 26(a), 
dual gradient calcualtion Equation 26(b), proximal function with respect to g 
26(c) and dual variable update 26(d).  

The main computational step in this algorithm is the dual gradient calcualtion. 
This is the Algorithm 1 in the paper and coded in the file Effinet_solve_Step.cuh.
The formualtion result in parallelisation across all the nodes at each stage and 
invovles a backward and forward traversal of the scenario tree. In this file, we 
coded these traversal with individual functions for backward substitution, 
forward substitution of the algorithm. 


As a final step, we compared the result from the APG algorith against the Gurobi 
result and check its tolerance. This tolerance depends on the value of the control -- 
when the value is greater then 1 percentage of the deviation is used and when it is 
less than 1 absoult deviation is used.

# Implementation details 
We use cuBLAS library for matrix-vector compuations. The solve-step is parallelisable 
stage-wise and implemented with the function cublasSgemmBatched function. We also defined 
_CUBLAS and _CUDA inline functions which use cudaError_t and cublasStatus_t for errors in
the routines of cuda and cublas.

 


