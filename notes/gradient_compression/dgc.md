
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

DGC naively perform fine-grained (i.e., element-wise) top-k to select gradients, and thus the communication will suffer from increased allgather data volume as #nodes increases.

CSC modified the process with coarse-grained sparsification: gradients are partioned into chunks, allreduce the gradient chunks selected based on allreduced L1-norm of each chunk, which gets rid of the allgather and solves the problem.


[deep-gradient-compression github](https://github.com/synxlin/deep-gradient-compression)


配置
```python
# configs/dgc/__init__.py
```

训练流程

```python
# train.py

from dgc.horovod.optimizer import DistributedOptimizer

# from dgc.compression import DGCCompressor
compression = configs.train.compression()
# cpr_parameters 即 dgc 处理的范围
compression.initialize(cpr_parameters.items())

# from dgc.optim import DGCSGD
optimizer = configs.train.optimizer(model.parameters())

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=configs.train.num_batches_per_step,
    op=hvd.Average
)

# 训练基本循环 zero_grad -> loss.backward -> optimizer.step
# 特别注意这里多次 backward 才走一次 step 更新
model.train()
for step, (inputs, targets) in enumerate(...):
    optimizer.zero_grad()

    # 这里用了内置循环累积梯度，比直接使用大 batch 剩显存
    # 注意这个 for 循环，对于 optimizer 里面理解 synchronize 过程非常重要
    for b in range(0, step_size, batch_size):
        _inputs = inputs[b:b+batch_size]
        _targets = targets[b:b+batch_size]
        _outputs = model(_inputs)
        _loss = criterion(_outputs, _targets)
        _loss.mul_(_r_num_batches_per_step)
        _loss.backward()
        loss += _loss.item()
    optimizer.step()
```

Optimizer

```python
# dgc/horovod/optimizer.py

class _DistributedOptimizer(torch.optim.Optimizer):

    def __init__(self, ...):
        # 初始化最后注册 通信 hook
        self._register_hooks()
        
    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    # 注册函数只执行一次，这里 zero grad 不是每次调用 hook
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    # 创建幽灵 tensor 来累积梯度，节点间同步，直至更新；
                    # p_tmp 和 p 使用同样的 storage，不占用额外显存
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    # 注册 _make_hook 这个关键 hook
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        # 这个 hook 有一个计数器，_allreduce_delay, 根据对象 p 不一样可以取不一样的值
        # 计数器不为零时跳过，这样可以让 grad 在本地累积，因为这个 hook 是做通信的
        # 效果为这个 hook 在多次调用才会被执行一次
        def hook(*ignore):
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    # 然后主要流程 step
    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)

    # step 调用 synchronize, 可以有跳过逻辑
    def synchronize(self):
        # 处理 hook 注册不成功，或者说 hook 没有被调用
        # hook 被调用后会添加 self._handles
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        # handle 为 None 的 hook 跳过又不跳过了？
        # 需要注意 synchronize 函数每个 step 被调用，但不是每次 backward 都会被调用
        # 在之前的 train 中有每个 step 会多次 backward，所以 grad 的 hook 会被多次调用，次数匹配
        # 所以代码执行到这里 handle 应该是一次调用_allreduce_grad_async 如果不是就补上
        for p, (handle, ctx) in self._handles.items():
            if handle is None:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)

        # for 循环处理异步通信的结果
        for p, (handle, ctx) in self._handles.items():
            output = self._synchronize_(handle)
            # 重置本地累积次数
            self._allreduce_delay[p] = self.backward_passes_per_step
            # 解压更新梯度
            p.grad.set_(self._compression.decompress(output, ctx))

        # 执行完毕，清理
        self._handles.clear()

    # 异步通信的 op，核心逻辑在 compression 中
    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor_compressed, ctx = self._compression.compress(p.grad, name)

        handle = self._communicate_(tensor_compressed, name=name, op=self.op)
        return handle, ctx

```

* hook 函数是一次注册，多次调用，所以 `self._handles` 会不断被填充，每次 synchronize 后可以被 clear

```python
# dgc/compression.py

class DGCCompressor:
    def __init__(self, ...):
        self.attributes = {}

    def initialize(self, named_parameters):
        # 工作范围
        for name, param in named_parameters:
            self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)

    def _sparsify(self, tensor, name):
        # 选出稀疏的 tensor 去通信
        # 原实现中比较复杂
        # 先随机选取部分梯度值的 TOPK 来计算阈值
        # 然后通过该阈值对原 tensor 做稀疏化
        importance = tensor.abs()
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()
        # 这里实现上有个 for 循环确保选出的 topk 满足要求
        indices = indices[:num_selects]
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            return tensor, ctx
        else:
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # 这里的 tensor 是个 tuple
            # decompress
            values, indices = tensor
            # 把同步回来的稀疏 tensor 对应位置更新
            # accumulate=True 处理 indices 中有重复的情况
            grad.zero_().index_put_([indices], values, accumulate=True)
            if self.op == Average:
                grad.mul_(1. / self.world_size)
            return grad.view(shape)
        else:
            return self.memory.compensate(tensor, name, accumulate=False)

    # optimizer _communicate_
    def communicate(self, tensor_compressed, name, op):
        # 两个分支
        if self.compress_ratio < 1.0 and name in self.attributes:
            # dgc 分支，tensor_compressed 是 tuple，各个节点选的 topk index 不相同
            # 所以使用 allgather 交换，然后各自解压、更新
            return [allgather_async_(t, name=f'{name}.t{e}')
                    for e, t in enumerate(tensor_compressed)]
        else:
            # 普通分支，直接 allreduce 完整 tensor
            return allreduce_async_(tensor_compressed, name=name, op=op)

    # optimizer _synchronize_
    def synchronize(self, handle):
        # from horovod.torch.mpi_ops import synchronize as synchronize_
        if isinstance(handle, (tuple, list)):
            return [synchronize_(h) for h in handle]
        else:
            return synchronize_(handle)
```

为了保证精度文章中介绍了下面几种补偿策略

* momentum correction
* local gradient clipping
* momentum factor masking, alleviate staleness
* warm-up training

前三种策略在 Memory 实现

```python
# dgc/memory.py

class DGCSGDMemory(Memory):

    def compensate(self, grad, name, accumulate=True):
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            # Momentum Correction
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(0, indices, 0)
        self.velocities[name].view(-1).index_fill_(0, indices, 0)
```
