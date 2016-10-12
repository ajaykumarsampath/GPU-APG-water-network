# Effinet-cuda
The Effinet-cuda repository containts the CUDA-C implementation of 
accelerated proximal gradient (APG) to solve the dual problem of the 
stocahstic model predictive optimisation problem for the management
of water networks of Barcelona as a part of the EFFINET project. 
Detailed explaination about theory and algorithm was discussed 
in the paper https://arxiv.org/abs/1604.01074. This paper is the bases 
for the functions defined in this package. 

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

 


