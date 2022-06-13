# Deep Gradient Compression

## Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training

Yujun Lin et. al. Tsinghua University, ICLR 2018

Gradient exchange require hight bandwidth,
* high latency
* low throughput
* poor connnection

Since
**99.9% of the gradient exchange in SGD are redundant,**
propose
**deep gradient compression (DGC)**
to 
**reduce communication bandwidth**.

To preserve accuracy,
* momentum correction
* local gradient clipping
* momentum factor masking, alleviate staleness
* warm-up training

improving local gradient accumulation and overcomming the staleness effect

DGC 
* pushes the gradient compression ratio to up to 600×
* not need to change the model structure
* no loss of accuracy

How

* only gradients larger than a threshold are transmitted
* accumulate the rest of the gradients locally, local gradient accumulation is equivalent to increasing the batch size over time

[deep-gradient-compression github](https://github.com/synxlin/deep-gradient-compression)

[GRACE](https://github.com/sands-lab/grace)

DGC naively perform fine-grained (i.e., element-wise) top-k to select gradients, and thus the communication will suffer from increased allgather data volume as #nodes increases.

CSC modified the process with coarse-grained sparsification: gradients are partioned into chunks, allreduce the gradient chunks selected based on allreduced L1-norm of each chunk, which gets rid of the allgather and solves the problem.

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

## 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs

Frank Seide et. al. MRA, Tsinghua, MR. INTERSPEECH 2014

1-BitSGDwithErrorFeedback

## GRACE: A Compressed Communication Framework for Distributed Machine Learning

Hang Xu et. al. 2021
