# Graph Convolutional Network (GCN)
GCN是提取拓扑图的空间特征的一种方式。拓扑图上难以直接使用CNN的原因是：CNN无法处理非欧式结构（Non Euclidean Structure）的数据，无法保持平移不变性（由于拓扑图中每个顶点相邻顶点数目不同）。

定义一个无向图 $G = (V, A)$ ， $V$ 代表图中节点的集合（ $|V| = N$ ）， $A \in R^{N \times N}$ 代表图中节点的邻接矩阵。 $D_{ii}=\sum_{j}A_{ij}$ 代表节点的度矩阵。

一个多层的图卷积网络如下：

$$
H^{(l + 1)} = \sigma(K H^{(l)} W^{(l)})
$$

其中， $K$ 是graph spectral filters， $H^{(l)}$ 是上一个图卷积层的输出， $W^{(l)}$ 是可学习的参数。

## 提取拓扑图空间特征的方式
- spatial domain：先确定receptive field（即如何寻找目标顶点的临近顶点），再确定使用何种方式处理包含不同数目顶点的特征。例如，[LC-RNN: A deep learning model for traffic speed prediction](https://www.ijcai.org/Proceedings/2018/0482.pdf)。
- **spectral domain：所谓的图卷积（GCN），这种思路就是借助图谱的理论来实现拓扑图上的卷积操作。首先从GSP（graph signal processing）中定义graph上的Fourier Transformation，进而定义graph上的convolution，最后与深度学习结合提出了图卷积（GCN）。例如，[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)**

## 图卷积（Graph Convolution）
根据卷积定理，卷积公式可以写成：

$$
f * g = \mathcal{F}^{-1} \lbrace \mathcal{F} \lbrace f \rbrace \odot  \mathcal{F} \lbrace g \rbrace \rbrace
$$

,其中 $\mathcal{F}$ 和 $\mathcal{F}^{-1}$ 分别表示傅立叶变换和傅立叶逆变换。

我们需要定义图上的傅立叶变换和傅立叶逆变换，来实现在图上的卷积。从Laplacian算子中可以找到傅立叶变换的基，其特征向量可以当作图上的傅立叶的基。对 $L$ 进行分解得到 $L = U \Lambda U^{-1} = U \Lambda U^{T}$ ，其中 $\Lambda$ 是特征值组成的对角矩阵。

| sdf | 传统Fourier变换 | Graph Fourier变换 |
|:---:|:---:|:---:|
| Fourier变换基| $e^{-2\pi ixv}$ | $U^\top$ |
| 逆Fourier变换基| $e^{2\pi ixv}$ | $U$ |
| Fourier变换基| $\infty$ | 点的个数 |

那么图上的傅立叶变换和傅立叶逆变换可以定义为：

$$
\mathcal{GF} \lbrace x \rbrace = U^T x \\
\mathcal{IGF} \lbrace x \rbrace = U x
$$

根据得到的图上的傅立叶变换 $\mathcal{GF}$ 和傅立叶逆变换 $\mathcal{IGF}$ ，可以得到图卷积（Graph Convolution）的公式：

$$
g * x = U(U^T g \odot U^T x)
$$

，可以将 $U^T g$ 看作是一个关于拉普拉斯特征值的函数 $g_{\theta}(\Lambda) = diag(\theta)$ ， $\theta \in R^N$ 代表可学习的参数，那么图卷积的公式还可以写成：

$$
g_{\theta} * x = U g_{\theta} U^T x = U g_{\theta}(\Lambda) U^T x 
$$

## 常见的拉普拉斯矩阵
- $L = D - A$ : Combinatorial Laplacian。
- $L^{sys} = D^{- \frac{1}{2}} L D^{-\frac{1}{2}}$ : Symmetric normalized Laplacian。标准化后的拉普拉斯矩阵，GCN中常用这种矩阵。
- $L^{rw} = D^{-1} L$ : Random walk normalized Laplacian。与Diffusion类似，使用到的论文有：[DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING](https://arxiv.org/pdf/1707.01926.pdf)


## 图卷积核参数的近似
因为图卷积的计算时需要与特征矩阵 $U$ 相乘，计算复杂度为 $O(N^2)$ ，当图非常大时，参数量大（ $g_{\theta}(\Lambda) = diag(\theta)$ ）且计算复杂度非常高。因此，需要采取一些方法避免直接与 $U$ 相乘，以减少计算量，甚至减少参数量。

### 多项式近似
将原本的 $g_{\theta}(\Lambda)$ 改写为多项式filter：

$$
g_{\theta}(\Lambda) = \sum^{K-1}_{k=0}{\theta_k \Lambda^k}
$$

其中， $\theta \in R^K$ 代表核可学习的参数，将原本的N个参数降低到K个，减少了参数量。而且，由于

$$U \sum^{K-1}_{k=0}{\theta_k \Lambda^k} U^T =  \sum^{K-1}_{k=0}{\theta_k L^k}$$

，所以，GCN的公式可以写为：

$$
g_{\theta} * x =  \sum^{K-1}_{k=0}{\theta_k L^k x}
$$

**优势**
- 卷积核的参数从N降低到K。
- 不需要进行特征分解了，直接使用L进行计算。
- 具有Local Connectivity。

### 切比雪夫多项式近似（Chebyshev Polynomials Approximation）
前一种多项式近似方法只是改变了核参数的计算方式，没有改变 $g_{\theta}$ 的计算方式，所以没有降低算法复杂度。可以利用切比雪夫多项式拟合卷积核（思想来自于GSP领域），降低算法复杂度和参数数量：

$$
g_{\theta}(\Lambda) =  \sum^{K-1}_{k=0}{\theta_k T_k(\tilde{\Lambda})}
$$

，其中 $\theta \in R^K$ 代表切比雪夫多项式的系数同时也是可学习的参数， $T_k(\tilde{\Lambda} \in R^{N \times N}$ 是切比雪夫多项式的第k项， $\tilde{\Lambda} = 2 \Lambda / \lambda_{max} - I_N$ 是标准化到[-1, 1]区间的特征值。切比雪夫多项式的计算方式为： $T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x)$ ，初始值 $T_0 = 1, T_1 = x$ 。

因为得到的 $g_{\theta}(\Lambda)$ 也是关于特征值 $\Lambda$ 的多项式，所以GCN的公式可以写成：

$$
g_{\theta} * x =  \sum^{K-1}_{k=0}{\theta_k T_k(\tilde{L}) x}
$$

，其中 $\tilde{L} = 2 L / \lambda_{max} - I_N$ 。

**优势**
- 具有多项式近似的优势。
- 降低了算法复杂度， $O(N^2)$ -> $O(K|\varepsilon|)$  。


### 一阶近似（1st-order Approximation）
相当于1阶切比雪夫多项式近似的特例，主要思想是更深的网络拟合效果优于更宽的网络。

令 $\lambda_{max} = 2$ ，因为网络训练中可以自动适应这个值，并且为了解决过拟合的问题和减少计算操作，令 $\theta = \theta_0 = -\theta_1$ ，得到GCN的公式：

$$\begin{aligned}
g_{\theta} * x &= \theta_0 x + \theta_1 (\tilde{L})x \\
&= \theta x - \theta (L - I_N)x \\
&= \theta x + \theta (D^{- \frac{1}{2}} A D^{-\frac{1}{2}})x \\
&= \theta (I_N + D^{- \frac{1}{2}} A D^{-\frac{1}{2}})x \\
&= \theta (\tilde{D}^{- \frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}})x
\end{aligned}$$

，其中 $\tilde{A} = A + I_n$ , $\tilde{D}_{ii} = \sum_{j}{\tilde{W}_{ij}}$ 。

# 推荐阅读
[Deep Learning with Graph-Structured Representations](https://pure.uva.nl/ws/files/46900201/Thesis.pdf)

# 参考
[1] [如何理解 Graph Convolutional Network（GCN）](https://www.zhihu.com/question/54504471/answer/332657604)
[2] [图卷积网络(GCN)新手村完全指南](https://zhuanlan.zhihu.com/p/54505069)
[3] [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
[4] [Spectral Networks and Deep Locally Connected Networks on Graphs](https://arxiv.org/pdf/1312.6203.pdf)
[5] [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf)
[6] [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/pdf/1709.04875.pdf)
