
## Optimizing Network Performance for Distributed DNN Training on GPU Clusters: ImageNet/AlexNet Training in 1.5 Minutes

Peng Sun et. al. SenseTime 2019

Communication backend: GradientFlow
* ring-based allreduce
* mixed-precision training
* computation/communication overlap
* lazy allreduce: fusing multiple communication operations
* coarse-grained sparse communication: only transmitting important gradient chunks

and also,

* momentum SGD correction
* warm-up dense training

DGC naively perform fine-grained (i.e., element-wise) top-k to select gradients, and thus the communication will suffer from increased allgather data volume as #nodes increases.

CSC modified the process with coarse-grained sparsification: gradients are partioned into chunks, allreduce the gradient chunks selected based on allreduced L1-norm of each chunk, which gets rid of the allgather and solves the problem.

