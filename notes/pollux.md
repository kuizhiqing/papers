# Pollux

## Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning

Aurick Qiao et al. Petuum, Inc, CMU, UCB, MBZUAI.

Award: Jay Lepreau Best Paper

集群调度的目标是为任务分配资源使得
* 训练时间小
* 集群资源利用率高
* 同时保证公平

一般的调度场景是提交任务时声明所需要的资源，
但是在深度学习任务中，资源分配之后如何高效地利用这些资源取决于如 batch size 和 learning rate 之类的配置，
这些配置一般是不属于调度器管理的。

由此，推出

Pollux: Co-adaptive Cluster Scheduler for DL

Pollux, 一个深度学习模型训练的互适应集群调度器。

它自动且动态地，
* 从整个集群性能和公平性的角度进行资源分配
* 动态地为每个训练任务调整 batch size 和 learning rate

什么是互适用呢？
* 从整个集群的角度做调度决策，同时在任务级别来调整配置参数
* 单个任务会根据分配到的资源进行动态地调整

结果：减少用户对任务的手动配置，缩短平均训练时常达 37-50%.

### 背景

#### 分布式-深度学习模型训练-数据并行

* mini-batch 组装 [batch size]
* 计算梯度 [计算密集]
* 梯度聚合 [通信密集]
* 更新参数 [learning rate]

#### batch size 对系统吞吐量的影响

总体而言，大 batch size 能够获得更高的吞吐，同时，这个关系

* 次/非线性 sub-linearly
* 一定量后，系统吞吐量不再随 gpu 数量提高

问题在与多少 gpu 合适呢？这取决于
* 模型结构
* 集群硬件
* batch size 等等

#### batch size 对实际效率（statistical efficiency）的影响

batch size 并不是越大越好，因为
* 需要同时调整 lr 确保模型质量（quality）[应该包括精度等指标]，而且很困难
* 增大 batch size 会降低训练的实际效率
* 大 batch size 训练的模型往往泛化能力较差

对于同样的数据吞吐量，大 batch size 所需要的训练步数较少（二者乘积一定），
但从模型实际效率上看，大 batch size 需要更多步数才能取得同样的效果。

#### 整体训练性能

经过以上分析可以看出，系统的吞吐和实际的训练效率相对于 batch size 存在经典的 trade-off 曲线关系，
所以，我们需要也可以找出他们的交叉点以获得优化的整体训练性能。

然而，这个优化的 batch size 在不同的训练阶段是不一样的（即随时间变化），最多能差 10 倍多 [McCandlish et al. 2018]，
即需要随训练进度动态地调整 batch size 以获得最优。

#### 集群调度
* 所需的 GPU 的取决于 batch size，反过来，合适的 batch size 取决于分配到的 GPU 资源
* 集群的 GPU 分配是一个全局的决策，需要考虑公平和竞争
* batch size 的选取还取决于具体的任务，如拓展性和实际效率等

用户在提交任务时决定上述配置是困难的，所以由集群调度器来调整这些配置参数可以优化深度学习模型的训练。

#### Pollux 集群调度器

* 计算模型训练性能
* 计算最优的资源需求、batch size、learning rate
* 重新调整资源分配，相应地调整任务的 batch size 和 learning rate

#### Key Idea: Goodput, not Throughput

Pollux 的优化目标是新定义的深度学习训练性能指标 goodput

$$
GOODPUT_t(a, m, s) = THROUGHTPUT(a, m, s) \times EFFICIENCY_t(M)
$$

其中，
* $a$ 资源分配向量，如 $a_n = #GPU$, 表示节点 n 上分配的 gpu 数
* $m$ 每个 gpu 的 batch size
* $s$ 梯度聚合步数
* $M$ 总 batch size，$M = |a|\times m\times s$

$(a, m, s)$ 会在训练过程中由 Pollux 自动计算, 
$THROUGHTPUT(a, m, s)$ 表示的是系统吞吐量（examples/second），
$EFFICIENCY_t(M)$ 表示实际训练效率（progress/example）.

#### 系统吞吐

$$
T_{iter}(a, m, s) = s\times T_{grad}(a, m) + (T_grad(a, m)^\gamma + T_{sync}(a)^\gamma)^{1/\gamma}
$$

其中，
* $T_{iter}$ 每步训练的时间
* $T_{grad}$ 计算梯度的时间
* $T_{sync}$ 网络通信的时间
* $s$ 梯度聚合的步数
* $\gamma$ 计算和通信重合度

Pollux 自动，
* 确定合适的 gpu 数量和 batch size
* 使用梯度聚合提高 batch size 达到 gpu 显存上限
* 把任务尽可能放置（pack）在尽量少的节点上以减少网络负载

#### 实际训练效率

每一个任务的实际训练效率可以表示为
$$
EFFICIENCY_t(M) = \frac{\phi_t + M_0}{\phi_t+M}
$$

其中，
* $M_0$ 表示用户定义的 baseline batch size
* $M$ 表示 batch size
* $\phi_t$ 梯度噪声 [McCandlish et al. 2018]

> 用户可以选择较小的初始 batch size $M_0$，Pollux 会选择不同的 bs 去平衡系统吞吐和实际训练效率。

关于梯度噪声 Gradient noise scale
* 较大的梯度噪声 -> 使用较大的 mini-batch 能够获得较高的实际效率
* 接近收敛的低信噪比 -> 更好的实际训练效率

Pollux 能够在不进行提前训练的情况下使用 $(a,m,s)$ 计算出任务的 GOODPUT.

#### 任务优化

在特定分配 gpu 为 a 的前提下，计算最优

$$
m^*, s^* = \operatorname{argmax}_{m,s} GOODPUT_t(a, m, s)
$$

改变 batch size 的同时，learning rate 也需要同步改变。
Pollux 为用户提供更新策略
* Linear scaling
* Square-root scaling
* AdaScale (Johnson et al. 2020)

#### 集群优化

优化目标

$$
FITNESS_p(A) = \left(\frac{1}{J} \sum_{j=1}^J SPEEDUP_j (A_j)^p\right)^{1/p}
$$

其中，

$$
SPEEDUP_j(A_j) = \frac{max_{m,s} GOODPUT_j(A_j, m_j, s)}{max_{m,s} GOODPUT_j(a_j, m_j, s)}
$$

p 是可变参数，用于控制任务间的公平性。

找到分配矩阵 $A$, $A_jn$ 表示节点 n 上分配给任务 j 的 gpu 数量。

* 对 A 的寻找使用 metaheuristic algorithm
* 调度要避免频繁的重新分配
* 避免分布式任务共享节点

#### Pollux 效果评估

Pollux 带来的主要收益是在共享集群上自动配置任务。

重点评估目标：即使任务已经给定理想的静态配置，Pollux 仍然能够相比于传统集群调度器有所提升。
包括以下方面，

* 真实的 Microsoft 深度学习分布式训练集群 (Jeon et al. 2019).
* 不同场景训练任务混合：图像分类、目标检测、语音识别、问答、推荐
* 手动配置 gpu 数量、batch size、learning rate、梯度聚合参数 (不使用Pollux，设定强baseline)

实验数据表明 Pollux 能比专家配置的任务缩短 37-50% 的平均训练时间。

#### 总结

* Pollux 同时从集群和任务的角度对任务参数进行优化
* Pollux 引入 goodput 概念，一种结合系统吞吐和实际效率的衡量标准
* Pollux 实测缩短 37-50% 的平均训练时间


