## 模型性能统计显著性（Statistical Significance）
### Paired Permutation Test
非参数检验：
[CS478 Paired Permutation Test Overview](https://axon.cs.byu.edu/Dan/478/assignments/permutation_test.php)

## 分布距离
### KL散度（Kullback-Leibler Deivergence）- 相对熵（Relative Entropy）
**相对熵**又被称为**KL散度**或者**信息散度（Information Divergence）**，是两个概率分布间差异的非对称性度量。在信息论中，相对熵等价于两个概率分布的信息熵差值，表示使用理论分布拟合真实分布时产生的信息损耗：
$$
\begin{align*}
D_{KL}(p||q) &= \sum^N_{i=1}[p(x_i){\rm log} p(x_i) - p(x_i){\rm log}q(x_i)] \\
&= \sum^N_{i=1}p(x_i){\rm log}\frac{p(x_i)}{q(x_i)}
\end{align*}
$$
，其中p是真实分布，q是理论（拟合）分布。

### 性质
#### 相对熵大于等于0
$$
\begin{align*}
D_{KL}(p||q) &= \sum^N_{i=1}p(x_i){\rm log}\frac{p(x_i)}{q(x_i)} \\
&= - \sum^N_{i=1}p(x_i){\rm log}\frac{q(x_i)}{p(x_i)} \\
&\geq - \sum^N_{i=1}p(x_i)(\frac{q(x_i)}{p(x_i)} - 1) \\
&= - \sum^N_{i=1}[q(x_i) - p(x_i)] \\
&= 0
\end{align*}
$$
，其中不等式由$log(x) \leq x - 1$推导得到，只在$p(x_i) = q(x_i)$时取等号。

#### 不对称
$$
D_{KL}(p||q) \neq D_{KL}(q||p)
$$
#### 不满足三角不等式
三角不等式：
$$
\begin{align*}
|x + y| &\leq |x| + |y| \\
|x - y| &\geq |x| - |y|
\end{align*}
$$
#### 参考
[如何简单易懂地理解变分推断(variational inference)](https://www.zhihu.com/question/41765860/answer/1149453776)
[进阶详解KL散度](https://zhuanlan.zhihu.com/p/372835186)

## 时序metric
#### 排列熵（Permutation Entropy）

https://www.aptech.com/blog/permutation-entropy/

#### RMSLE （Root Mean Squared Logarithmic Error）
惩罚"欠预测"大于"过预测": log函数在值小时的变化大于值大时的变化。
$$
RMSLE = \sqrt{\frac{1}{n}\sum^{n}_{i=1}({log(\hat{y}_i + 1) - log(y_i + 1)})^2}
$$
The smaller the better.
