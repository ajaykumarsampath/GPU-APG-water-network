# GPU-APG-Water-network 
The objective of this project is to manage the water network taking the 
stochastic nature of the water demand. We proposed to use stochastic model 
predictive controller that considers the probabilistic nature of the 
water demand in deciding the control policy. We identified the main hinderance 
for practical implementation of the stochastic predictive control in water 
network is lack of algorithms that can solve the control problem in real-time. 
To overcome this limitation we proposed an optimisation algotithm that exploits
the structure of a stochastic optimal control problem and suitable for parallelisation. 
This method use the dual formulation and uses the accelerated proximal gradient 
(APG) algorithm to solve this. Detailed explaination about theory and 
algorithm is discussed in the paper https://arxiv.org/abs/1604.01074.


This repository containts the CUDA-C implementation of the accelerated proximal
gradient (APG) algorithm to solve the stochastic predicitve control problem on
the water networks of Barcelona. This work is done as a part of the EFFINET project. 
The routines defined are based on the Algorithms in the paper.

# Explanation details 
The file main_effinet.cu contains the main function. The list of 
functions is given in the file api_effinet_cuda.cuh. Before executing the 
actual algorithm, memory for all the variables involved is allocated 
in GPU. This include the system state, control, system dynamic equations,
tree strucute, dual variables and all the off-line matrices. All matrix
multipliactions are perfomed used cuBLAS library. 

The steps of the algorithm APG algorithm is given by Equation 26 of the
paper. This is implemented in the file Effinet_APG.cuh. For each step in 
the algorithm, we defined a function -- for the extration step Equation 26(a), 
dual gradient calcualtion Equation 26(b), proximal function with respect to g 
26(c) and dual variable update 26(d). All the off-line matrics are given by the 
Appendix B in the paper. These calculations are implemented in the file 
Effinet_factor_step.cuh. Note that these calculation need to be preformed before
the actual APG algorithm. These off-line calcualtion depend on the tree structure, 
system dynamics and cost function. As long as they are constant, the resulting 
off-line matrices need not be changed. 


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

# Implmentation details 
We use cuBLAS library for matrix-vector compuations. The solve-step is parallelisable 
stage-wise and implemented with the function cublasSgemmBatched function. We also defined 
_CUBLAS and _CUDA inline functions which use cudaError_t and cublasStatus_t for errors in
the routines of cuda and cublas.

 


