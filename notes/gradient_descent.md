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

