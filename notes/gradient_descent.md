# Gradient Descent

## An overview of gradient descent optimization algorithms

Sebastian Ruder, Insight Centre for Data Analytics, NUI Galway Aylien Ltd., Dublin, 2017

#### Gradient Descent

Multi-variable function $f : \mathbb{R}^n \mapsto \mathbb{R}$, 
defined differentiable in a neighborhood of a point $x\in\mathbb{R}^n$,
for $\lambda \in\mathbb{R}_+$ small enough,

$$
x_{k+1} = x_k -\lambda \nabla f(x_k) 
$$

leads to $f(x_{k+1})\le f(x_k)$.

If $f$ convex and $\nabla f$ Lipschitz, $f_k$ converge to a local mimimum.

#### Optimization : Momentum

Let $\gamma <1,\, v_0 = 0$,

$$
\begin{cases}
x_{k+1} &= x_k - v_k \\
v_k &= \gamma v_{k-1} + \lambda \nabla f(x_k)
\end{cases}
$$

The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.

#### Optimization : Nesterov

Version with correction,

$$
\begin{cases}
x_{k+1} &= x_k - v_k \\
v_k &= \gamma v_{k-1} + \lambda \nabla f(x_k - \gamma v_{k-1})
\end{cases}
$$

This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks.

#### Adagrad

It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing with sparse data.

$$
x_{k+1} = x_k - \frac{\lambda}{\sqrt{G_k+\epsilon}} \nabla f(x_k)
$$

Application : learned to recognize cats in Youtube videos; GloVe word embeddings.

#### Adadelta

Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.

#### RMSprop

RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad’s radically diminishing learning rates. RMSprop in fact is identical to the first update vector of Adadelta.

$$
v(x_k,t):=\gamma v(x_k,t-1)+(1-\gamma )(\nabla f(x_k))^{2}
$$

$$
x_{k+1} = x_k -{\frac {\eta }{\sqrt {v(x_k,t)}}}\nabla f(x_k)
$$

#### Adam

Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. 
In addition to storing an exponentially decaying average of past squared gradients $v_k$ like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients $u_k$, similar to momentum.

$$
\begin{aligned}
u_k &= \beta_1 u_{k-1} + (1-\beta_1)\nabla f(x_k) \\
v_k &= \beta_2 v_{k-1} + (1-\beta_2)\nabla^2 f(x_k) \\
\end{aligned}
$$

$$
\begin{aligned}
\hat u_k &= \frac{u_k}{1-\beta_1} \\
\hat v_k &= \frac{v_k}{1-\beta_2}
\end{aligned}
$$

$$
x_{k+1} = x_k -{\frac {\eta }{\sqrt {\hat v_k} + \epsilon }}\nabla f(x_k)
$$

## PyTorch Implementation

#### SGD, Momentum

The implementation of SGD with Momentum/Nesterov subtly differs from
Sutskever et. al. and implementations in some other frameworks.

Considering the specific case of Momentum, the update can be written as

$$
    \begin{aligned}
        v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
        p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
    \end{aligned}
$$

where $p$, $g$, $v$ and $\mu$ denote the
parameters, gradient, velocity, and momentum respectively.

This is in contrast to Sutskever et. al. and
other frameworks which employ an update of the form

$$
    \begin{aligned}
        v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
        p_{t+1} & = p_{t} - v_{t+1}.
    \end{aligned}
$$

The Nesterov version is analogously modified.

```python
# torch/optim/sgd.py

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)

```

#### Adagrad

Algorithm

$$
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  statesum_0 \leftarrow 0                             \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}statesum_t  \leftarrow  statesum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{statesum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
$$

```python
# torch/optim/adagrad.py

def _single_tensor_adagrad(params: List[Tensor],
                           grads: List[Tensor],
                           state_sums: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           lr: float,
                           weight_decay: float,
                           lr_decay: float,
                           eps: float,
                           has_sparse_grad: bool):

    for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
        # update step
        step_t += 1
        step = step_t.item()

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
        else:
            is_complex = torch.is_complex(param)
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)

```

#### Adam

Algorithm

$$
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
$$


```python
# torch/optim/adam.py

def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1
        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)



        step_size = lr / bias_correction1
        # param = param - step_size * (exp_avg / denom)
        # element-wise division
        param.addcdiv_(exp_avg, denom, value=-step_size)

```

AdamW is Adam with correct Weight Decay, when weight decay is 0, there is no difference between Adam and AdamW.
