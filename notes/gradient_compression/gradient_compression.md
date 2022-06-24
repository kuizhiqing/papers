
## 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs

Frank Seide et. al. MRA, Tsinghua, MR. INTERSPEECH 2014

1-BitSGDwithErrorFeedback

总结：1 bit 量化的思路是把每个 32 位的值按照正负分别用 1 或 0 量化，然后通信，更新时 1 则更新 +1，0 则更新 -1，同时本地量化差异保留补偿。这样的效果应该相当于本地梯度累积，每次通信根据正负同步 1 个单位的量。当然，每次更新带上系数，最终得到收敛效果。

## 1-bit Adam: Communication Efficient Large-Scale Training with Adam’s Convergence Speed

Hanlin Tang et. al. Microsoft 2021

总结：adam 的梯度更新是非线性的，沿用 1bit sgd 的梯度累积或者说梯度补偿策略无法保证收敛性，文章提出了针对 adam 的保证收敛性的 1 bit 梯度量化方法并给出了理论证明。方法的核心在于在 warm-up 阶段计算 momentum 方差，用于后面进行误差补偿的计算。

## Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training

Yujun Lin et. al. Tsinghua University, ICLR 2018

[DGC](https://github.com/synxlin/deep-gradient-compression)


## Optimizing Network Performance for Distributed DNN Training on GPU Clusters: ImageNet/AlexNet Training in 1.5 Minutes

Peng Sun et. al. SenseTime 2019


## GRACE: A Compressed Communication Framework for Distributed Machine Learning

Hang Xu et. al. 2021

[GRACE](https://github.com/sands-lab/grace)


## SIDCo An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems

Ahmed M. Abdelmoniem et. al. CEMSE, KAUST. MLSys 2021

[SIDCo](https://github.com/sands-lab/SIDCo)

## PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization

Thijs Vogels et. al. EPFL. NeurIPS 2019

[PowerSGD](https://github.com/epfml/powersgd)

## Don't Use Large Mini-Batches, Use Local SGD

Tao Lin et. al.  EPFL. ICLR 2020

[LocalSGD-Code](https://github.com/epfml/LocalSGD-Code)

总结：为减轻同步模式中慢节点的影响，可以减少通信，这会带来精度损失。使用 local SGD 的方法可以现在节点内进行 SGD 更新，多步之后再同步各个节点上的参数。 post local SGD 的方法将训练过程分成两阶段：先使用同步 SGD，再增大同步间隔提高训练吞吐。

## Adaptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD

Jianyu Wang et. al. Carnegie Mellon University. SysML 2019

总结：动态调整参数同步间隔来平衡训练吞吐和精度。


## Overlap Local-SGD: An Algorithmic Approach to Hide Communication Delays in Distributed SGD

Jianyu Wang et. al. ICASSP 2020

[Overlap_Local_SGD](https://github.com/JYWa/Overlap_Local_SGD)
