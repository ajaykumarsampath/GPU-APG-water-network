# GPU-accelerated stochastic MPC

## About
The objective of this project is to incorporate the stochastic nature of 
the water demand for economical managment the water networks. 
A stochastic model predictive controller
considers the probabilistic nature of the water demand in deciding an optimal control
policy. But the main hinderance for the practical implementation of the stochastic
predictive control in water networks is the lack of proper optimisation algorithms
that can solve this problem in real-time. To overcome this limitation we 
propose an optimisation algotithm that exploits the structure of stochastic
optimal control problems and is amenable to parallelisation. This method is 
based on accelerated proximal gradient (APG) algorithm and uses the dual
formulation to solve the problem. Theoretical details are discussed in the paper 
https://arxiv.org/abs/1604.01074.

This repository containts the CUDA-C implementation of the accelerated proximal
gradient (APG) algorithm to solve the stochastic predicitve control problem 
for the operational management of the water network of Barcelona. 
This work was financially supported by the EU FP7 research
project EFFINET "Efficient Integrated Real-time monitoring
and Control of Drinking Water Networks," grant agreement
no. 318556. The routines defined here are based on the Algorithms in the paper.


## Details
The file `main_effinet.cu` contains the main function. The API of this 
implementation is given in the file `api_effinet_cuda.cuh`. Before executing the 
actual algorithm, memory for all the variables involved is allocated 
on GPU. This includes the system state, control, system matrices,
tree strucute, parameters of the cost function and constraints on state and control input. 
All the computations that re involved in the APG algorithm are matrice-vector 
operations. The matrices that are involved depend only on the 
system dynamics, constraints, cost function and tree structure. Therefore, 
these matrices need not be calculated at every sampling time (details can be 
found in Appendix B in the above paper). These matrices are implemented in the file 
Effinet_factor_step.cuh. Note that these calculation need to be preformed before
the actual APG algorithm and need not be accounted in the actual runtime
of the algorithm. 

The steps of the algorithm APG algorithm is given by Equation 26 of the
paper. This is implemented in the file `Effinet_APG.cuh`. For each step in 
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

## Implementation details 
We use cuBLAS library for matrix-vector compuations. The solve-step is parallelisable 
stage-wise and implemented with the function cublasSgemmBatched function. We also defined 
`_CUBLAS` and `_CUDA` inline functions which use `cudaError_t` and `cublasStatus_t` for errors in
the routines of cuda and cublas.


## Licence
This is a free software and is distributed with the terms of the
[LGPL v3](https://github.com/ajaykumarsampath/GPU-APG-water-network/blob/master/LICENCE.txt) licence.

## References

* A.K. Sampathirao, P. Sopasakis, A. Bemporad and P. Patrinos,
  _GPU-accelerated stochastic predictive control of
  drinking water networks_, https://arxiv.org/abs/1604.01074 (submitted for
  publication to IEEE CST), 2016.
 


