# Long- and Short-term Time-series Networks (LSTNet)
端到端神经网络，点预测，支持多步预测，多元时间序列预测。
Published in SIGIR 2018.

## Notations
**模型输入**：前 $T$ 个时刻的多元时间序列的观测值。
$$
X_T = \lbrace y_1, y_2, \cdots, y_T \rbrace \in R^{n \times T}
$$
，其中 $y_t \in R^n$ 表示第 $t$ 时刻的所有时间序列的观测值，$n$ 表示时间序列的数量。

**模型输出**：未来第 $T+h$ 时刻的多元时间序列的预测值。
$$
y_{T+h}  \in R^{n}
$$
，其中 $h$ 表示任务需要预测的时刻的horizon。

## Model Architecture
模型主要包含以下模块：
- CNN：用于捕获多元变量之间的局部相关性和短时相关性。
- RNN + Att（或Skip-RNN）：用于捕获时间序列的长时相关性。
- AR：用于解决神经网络模型的scale insensitive问题。

### Convolutional Component (CNN)
使用没有Polling层的卷积同时在时间维度和多变量维度上提取变量（多元时间序列）之间的短时相关性和局部相关性：
$$
h_k = RELU(W_k * X_T + b_k) \in R^T
$$
，其中 $*$ 表示卷积操作，$h_k \in R^T$ 表示第$k$个filter的卷积输出，通过在$X_T$前填充零值得到$T$个输出。

经过CNN层后，得到：
$$
H_C = \lbrace h_1, h_2, \cdots, h_{d_c} \rbrace \in R^{T \times d_c}
$$
，其中 $d_c$ 表示卷积核的数量。为了后续使用，可以CNN的输出也可以表示为：
 $$
H_C^\top = \lbrace h^{c}_1, h^{c}_2, \cdots, h^{c}_{T} \rbrace \in R^{d_c \times T}
 $$

### Recurrent Component (RNN)

通常来说，时间序列特征分为临近性（closeness）、周期性（period）和趋势性（trend）特征三部分。
#### Closeness
使用RNN在时间维度上去捕获多元时间序列的临近特征：
$$
h^{R}_{t} = f_{R}(h^{c}_t, h^{R}_{t-1}) \in R^m
$$
，其中$h^{c}_t$是CNN层的输出，$f_{R}$是激活函数为ReLU的GRU或者LSTM，$h^{R}_{t-1}$是在第$t-1$时刻的输出状态，$m$是GRU或者LSTM的隐藏单元数。

#### Period

##### Recurrent-skip Component
对于较长的时间序列，即便是LSTM也难以捕获长时特征，因此本文提出了一种跳跃RNN，用于捕获多元时间序列的长时特征：
$$
h^{S}_{t} = f_{S}(h^{c}_t, h^{S}_{t-p}) \in R^m
$$
，其中$h^{c}_t$是CNN层的输出，$f_{S}$是激活函数为ReLU的GRU或者LSTM，$h^{S}_{t-p}$是在第$t-p$时刻的输出状态，$m$是GRU或者LSTM的隐藏单元数，$p$表示数据的一个周期的时间间隔数（先验知识）。

此时，神经网络部分的输出为：
$$
h^D_t = W^R h^R_t + \sum_{i=1}^{p}{W_i^S h^S_{t-i} + b} \in R^n
$$
，其中$h^D_t$是模型神经网络部分在第$t$时刻的输出。

##### Temporal Attention Layer
由于上一节提到的跳跃RNN需要预先定义周期p，对于一些周期不明显的数据集很难定义p。为了解决这个问题，可以使用注意力机制在时间维度上动态捕获相关的隐状态：
$$
\alpha_t = AttnScore(H_t^R, h^R_{t}) \in R^q \\
c_t = H^R_t \alpha_t \in R^m
$$
，其中$H^R_t = \lbrace h^R_{t-q}, \cdots, h^R_{t-1} \rbrace \in R^{m \times q}$，$AttnScore$是通用的注意力机制的方法（点乘、加和等），$c_t$表示长时历史特征对当前时刻的相关性加权和的状态。

此时，神经网络部分的输出为：
$$
h^D_t = W [c_t;h^R_t] + b \in R^n
$$
，其中$h^D_t$是模型神经网络部分在第$t$时刻的输出。

### Autoregressive Component (AR)
由于卷积网络模块和循环神经网络模块的非线性特性，神经网络模型的一个主要缺点是输出的scale对输入的scale不敏感。在特定的真实数据集中，输入数据的大小以非周期性的方式不断变化，这大大降低了神经网络模型的预测精度。

为了解决这个问题，借鉴了highway network的想法，引入经典的自回归模型作为线性部分：
$$
h^L_{t,i} = \sum^{q^{ar}-1}_{k=0}{W^{ar}_k y_{t-k,i} + b^{ar}} \in R
$$
，其中$q_{ar}$表示ar模块的观测窗口长度，$h^L_{t,i}$是$h^L_{t} \in R^n$的第$i$个值，$y_{t-k,i}$是$y_{t-k} \in R^n$的第$i$个值。

最终，LSTNet模型的输出为神经网络部分和线性模块部分的输出的加和：
$$
\hat{Y}_t = h^D_t + h^L_t \in R^n
$$

### Objective Function 
$$
\substack{minimize \\ \Theta} \sum_{t \in \Omega_{train}} \sum_{i=0}^{n-1}{|Y_{t,i} - \hat{Y}_{t,i}|}
$$

## Experiment

### Metrics
- Root Relative Squared Error (RSE)
$$
RSE = \frac{\sqrt{\sum_{(i,t) \in \Omega_{test}}(Y_{i,t} - \hat{Y}_{i,t})^2}}{\sqrt{\sum_{(i,t) \in \Omega_{test}} (Y_{i,t} - mean(Y))^2}}
$$

- Empirical Correlation Coefficient (CORR)
$$
CORR = \frac{1}{n} \sum_{i=1}^{n}{ \frac{\sum_{t}(Y_{i,t} - mean(Y_i)) (\hat{Y}_{i,t} - mean(\hat{Y}_i))}{\sqrt{\sum_{t} (Y_{i,t} - mean(Y_i))^2 (\hat{Y}_{i,t} - mean(\hat{Y}_i))^2}}}
$$

### Results
详见论文。

# 参考
[1] [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/pdf/1703.07015.pdf)