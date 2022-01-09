## Neural ODE
### Oridinary Differential Equation (ODE)
残差网络（residual networks）的形式：
$$
h_{t+1} = h_t + f(h_t, \theta_t)
$$
使用更多层网络，更小的时间步，则可以转化为常微分方程形式：
$$
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)
$$
，通常已知初始状态$\mathbf{z}_0 = \mathbf{z}(t=0) \in R^D$。则$\mathbf{z}(t)$可以通过积分得到：
$$
\mathbf{z}(t) = \mathbf{z}(0) + \int_0^t{f(\mathbf{z}(t'), t', \theta) dt'}
$$
计算积分的实现比较复杂，我们可以使用一些方法得到积分的近似解，比如一种最简单的方式，对极小的时间间隔($\triangle t$)的离散值进行加和来代替积分：
$$
\mathbf{z}(t) = \mathbf{z}(N\triangle t) \approx \mathbf{z}(0) + \triangle t \sum_{k=0}^{N-1}{f(\mathbf{z}(k\triangle t), k\triangle t, \theta)}
$$
，省略参数$\theta$和时间，使用一种更简洁的方式表达：
$$
\mathbf{z}(t) = \mathbf{z}_N \approx \mathbf{z}_0 + \triangle t \sum_{k=0}^{N-1}{f(\mathbf{z}_k)} = \mathbf{z}_{N-1} + \triangle t f(\mathbf{z}_{N-1})
$$
像这样计算积分的方法就叫作ODE Solver。以上这种简单的方法叫Euler Method，精度较低且不稳定，其他更优的方法还有不同阶的Runge-Kutta Method等。
上述公式用另一种更通用的方式表达：
$$
\mathbf{z}(t_1) = ODESolver(\mathbf{z}(t_0), f, t_0, t_1, \theta)
$$
部分代码如下：
```python
def rk4_step_method(diff_func, dt, z0):
    """
    Fourth order Runge-Kutta method for solving ODEs.

    Args:
        diff_func: function(dt, state)
            Differential equation.
        dt: Tensor with shape [..., 1]
            Equal to t1 - t0.
        z0: Tensor with shape [..., dim]
            State at t0.

    Returns:
        Tensor with shape [..., dim], which is updated state.
    """
    # shape -> [..., dim]
    k1 = diff_func(z0)
    k2 = diff_func(ode_update_op(z0, k1, dt / 2.0))
    k3 = diff_func(ode_update_op(z0, k2, dt / 2.0))
    k4 = diff_func(ode_update_op(z0, k3, dt))

    if isinstance(z0, list) or isinstance(z0, tuple):
        return [item_z + (item_k1 + 2.0 * item_k2 + 2.0 * item_k3 + item_k4) * dt / 6.0
                for item_z, item_k1, item_k2, item_k3, item_k4 in zip(z0, k1, k2, k3, k4)]
    else:
        return z0 + dt * (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0
 
def neural_ode_layer(
        z0,
        dt,
        ode_func
):
    """
    Calculate z1 by z0 and time gap dt.

    Args:
        z0: Tensor with shape [..., dim]
        dt: Tensor with shape [..., 1 or dim]

    Returns:
        A tensor presents z1, whose shape is the same as z0.
    """
    from .utils import prod
    with tf.name_scope('neural_ode'):
        # Forward activity
        dt_ratio = 1.0 / num_samples
        delta_t = dt * dt_ratio

        z = z0
        z_list = []
        for i in range(num_samples):
            z = rk4_step_method(ode_func, delta_t, z)
            z_list.append(z)
        z1 = z
    return z1
```

### ODE与神经网络的联系
Residual network、RNN以及normalizing flow等都是要在hidden state上建立如下变换：
$$
\mathbf{z}_{t+1} = \mathbf{z}_t + f(\mathbf{z}_t, \theta_t)
$$
当使用更多层数（$f$函数）和更小的时间步（$t \sim t+1$之间增加更多的时间步）时，那么如上迭代式更新方式可以看做是连续变换的欧拉离散化，我们可以使用一个神经网络化的ODE来表示：
$$
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)
$$

### Neural ODE的优势
- 内存开销小：在反向传播时，使用ode solver求解不需要存储中间梯度，无论使用多少层，都能够保证常数级的内存开销。
- 模型精度灵活：模型能够根据问题的复杂度，自由的调整精度。也能够在训练时使用高精度的模型时，在预测时使用低精度的模型（比如实时任务）
- 能够对连续时间建模

### Neural ODE的优化
我们将ODE Solver当作一个黑盒，那么可以使用伴随方程（adjoint sensitivity method）进行梯度计算。假设时间是从$t_0$到$t_1$，优化neural ode层需要计算loss对状态$z_0$的梯度$a(t_0)=\frac{\partial L}{\partial z(t_0)}$、对模型参数$\theta$的梯度$a_\theta(t_0)= \frac{dL}{d\theta(t_0)}$，这两部分我们可以使用如下公式进行计算：
首先计算这两者对$t$的导数：
$$
\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f(z(t),t,\theta)}{\partial z} \\
\frac{da_\theta(t)}{dt} = -a(t)^\top \frac{\partial f(z(t),t,\theta)}{\partial \theta} \\
$$
，由于再通过积分计算可以得到这两者：
$$
a(t_0)=\frac{\partial L}{\partial z(t_0)} = a(t_1) + \int_{t_1}^{t_0}-a(t)^\top \frac{\partial f(z(t),t,\theta)}{\partial z}dt \\
a_\theta(t_0) = \frac{dL}{d\theta(t_0)} = a_\theta(t_1) + \int_{t_1}^{t_0}-a(t)^\top \frac{\partial f(z(t),t,\theta)}{\partial \theta}dt \\
$$
，其中，$a(t_1)$是可以通过neural ode layer外的BP得到，设置$a_\theta(t_1)$为0，那么这两者就可以通过ode solver进行求解。
部分代码如下：
```python
def grad(a1, variables=None):
    # a1 is grad_z1 == dL/dz1
    if variables is None:
        variables = []

    def aug_dynamics(tmp_states):
        """
        Ode function for states [z_1, a_1, \thetas (many)].

        Args:
            tmp_states: list
                Elements are [z_1, a_1, \thetas (many)].

        Returns:
            List contains differentiations of states.
        """

        tmp_z = tmp_states[0]
        tmp_neg_a = -tmp_states[1]

        # using GradientType to calculate (faster when building graph)
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch([tmp_z, *variables])
            res_dz = ode_func(tmp_z)
            tmp_ds = g.gradient(res_dz, [tmp_z, *variables], output_gradients=tmp_neg_a)

        res_da = tmp_ds[0]
        res_dtheta = [flat_tensor(var, num_last_dims=len(var.get_shape().as_list()))
                      for var in tmp_ds[1:]]

        return [res_dz, res_da, *res_dtheta]

    init_var_grad = [tf.zeros([prod(var.get_shape().as_list())]) for var in variables]

    if a1 is None:
        a1 = tf.zeros_like(z1)

    # [z(t_1), a(t_1), \theta]
    states = [z1, a1, *init_var_grad]
    # print('states:', states)
    for i in range(num_samples):
        states = method(aug_dynamics, -delta_t, states)

    grad_z0 = states[1]
    grad_t = tf.ones_like(dt)

    # average the different dt effect on variable \theta
    grad_theta = [tf.reshape(tf.reduce_mean(var_grad, axis=0), var.shape) for var, var_grad in
                  zip(variables, states[2:])]
    return (grad_z0, grad_t), grad_theta
```

### 完整Neural ODE Layer的tf1代码
**ode_func包含的变量必须为ResourceVariable才可以使用GradientTape**
```python
def get_neural_ode_layer(
        ode_func,
        num_samples=10,
        method=rk4_step_method,
        return_states=False
):
    """
    Get a black-box neural ode layer parameterized by parameters.

    Args:
        ode_func: function
            It likes f(solver_function, dt, z_list), and contains the learnable variables.
        num_samples: int
            Number of samples in time interval dt.
        method: function
            Solver function like f(ode_func, dt, z_list)
        return_states: bool, default False
            Identify whether return whole states or just last state.

    Returns:
        A neural_ode_layer (function) with signature f(z0, dt).
    """

    @tf.custom_gradient
    def neural_ode_layer(
            z0,
            dt,
    ):
        """
        Calculate z1 by z0 and time gap dt.

        Args:
            z0: Tensor with shape [..., dim]
            dt: Tensor with shape [..., 1 or dim]

        Returns:
            A tensor presents z1, whose shape is the same as z0.
        """
        from .utils import prod
        with tf.name_scope('neural_ode'):
            # Forward activity
            dt_ratio = 1.0 / num_samples
            delta_t = dt * dt_ratio

            z = z0
            z_list = []
            for i in range(num_samples):
                z = method(ode_func, delta_t, z)
                z_list.append(z)
            z1 = z

            def grad(a1, variables=None):
                # a1 is grad_z1 == dL/dz1
                if variables is None:
                    variables = []

                def aug_dynamics(tmp_states):
                    """
                    Ode function for states [z_1, a_1, \thetas (many)].

                    Args:
                        tmp_states: list
                            Elements are [z_1, a_1, \thetas (many)].

                    Returns:
                        List contains differentiations of states.
                    """

                    tmp_z = tmp_states[0]
                    tmp_neg_a = -tmp_states[1]
                    # tmp_var_grad = tmp_states[2:]

                    # calculate dz/dt

                    # using tf.gradients to calculate
                    # res_dz = ode_func(tmp_z)
                    # tmp_ds = tf.gradients(res_dz, [tmp_z, *variables], grad_ys=tmp_neg_a)

                    # or using GradientType to calculate (faster when building graph)
                    with tf.GradientTape(watch_accessed_variables=False) as g:
                        g.watch([tmp_z, *variables])
                        res_dz = ode_func(tmp_z)
                        tmp_ds = g.gradient(res_dz, [tmp_z, *variables], output_gradients=tmp_neg_a)

                    res_da = tmp_ds[0]
                    res_dtheta = [flat_tensor(var, num_last_dims=len(var.get_shape().as_list()))
                                  for var in tmp_ds[1:]]

                    return [res_dz, res_da, *res_dtheta]

                # Backward activity
                # Compile EAGER graph to static (this will be much faster)
                import tensorflow.contrib.eager as tfe
                aug_dynamics = tfe.defun(aug_dynamics)

                # Construct back-state for ode solver
                # reshape variable \theta for batch solving
                init_var_grad = [tf.zeros([prod(var.get_shape().as_list())]) for var in variables]

                if a1 is None:
                    a1 = tf.zeros_like(z1)

                # [z(t_1), a(t_1), \theta]
                states = [z1, a1, *init_var_grad]
                # print('states:', states)
                for i in range(num_samples):
                    states = method(aug_dynamics, -delta_t, states)

                grad_z0 = states[1]
                grad_t = tf.ones_like(dt)

                if variables is not None:
                    # average the different dt effect on variable \theta
                    grad_theta = [tf.reshape(tf.reduce_mean(var_grad, axis=0), var.shape) for var, var_grad in
                                  zip(variables, states[2:])]
                    return (grad_z0, grad_t), grad_theta
                else:
                    return grad_z0, grad_t

        if return_states:
            return z_list, grad
        else:
            return z1, grad

    return neural_ode_layer
```


# 参考
【1】[Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf)