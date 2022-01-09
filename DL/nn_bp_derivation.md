### 单层神经网络的Back Propagation Through Time (BPTT)计算
假设
$$
C = A B
$$
，其中$C \in R^{3 \times 3}$，$A \in R^{3 \times 4}$，$B \in R^{4 \times 3}$，值分别如下：
$$
\begin{bmatrix}
14 & 38 & 62 \\
38 & 126 & 214 \\
62 & 214 & 366 
\end{bmatrix} =
\begin{bmatrix}
0 & 1 & 2 & 3 \\
4 & 5 & 6 & 7 \\
8 & 9 & 10 & 11 
\end{bmatrix} \times
\begin{bmatrix}
0 & 4 & 8 \\
1 & 5 & 9 \\
2 & 6 & 10 \\
3 & 7 & 11 
\end{bmatrix}
$$

此时，求梯度$\frac{\partial C}{\partial A} \in R^{3 \times 4}$，矩阵shape与A的shape一致，具体如下：

$$
\frac{\partial C}{\partial A}

=
\begin{bmatrix}
\frac{\partial C}{\partial a_{11}} & \frac{\partial C}{\partial a_{12}} & \frac{\partial C}{\partial a_{13}} & \frac{\partial C}{\partial a_{14}} \\
\frac{\partial C}{\partial a_{21}} & \frac{\partial C}{\partial a_{22}} & \frac{\partial C}{\partial a_{23}} & \frac{\partial C}{\partial a_{24}} \\
\frac{\partial C}{\partial a_{31}} & \frac{\partial C}{\partial a_{32}} & \frac{\partial C}{\partial a_{33}} & \frac{\partial C}{\partial a_{34}}
\end{bmatrix}
$$

，其中$a_{ij}$是标量scalar，$C$是矩阵，矩阵对标量的偏导数计算如下：

$$
\frac{\partial C}{\partial a_{11}} = \sum_i^3 \sum_j^3 \frac{\partial c_{ij}}{\partial a_{11}}
$$

，也就是偏导矩阵的所有元素加和，我们如此定义：

$$
\frac{\partial C}{\partial a_{11}} =
\sum

\begin{bmatrix}
\frac{\partial c_{11}}{\partial a_{11}} & \frac{\partial c_{12}}{\partial a_{11}} & \frac{\partial c_{13}}{\partial a_{11}} \\
\frac{\partial c_{21}}{\partial a_{11}} & \frac{\partial c_{22}}{\partial a_{11}} & \frac{\partial c_{23}}{\partial a_{11}} \\
\frac{\partial c_{31}}{\partial a_{11}} & \frac{\partial c_{32}}{\partial a_{11}} & \frac{\partial c_{33}}{\partial a_{11}}
\end{bmatrix} = 
\sum
\begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}

= b_{11} + b_{12} + b_{13} = 12
$$
，如此计算每一个C对A的偏导，就得到$\frac{\partial C}{\partial A}$。


### 多层神经网络的BPTT计算（Chain Rule）
承接上述单层神经网络的梯度计算，假设loss或者上一层的输出对C的梯度为:
$$
\frac{\partial L}{\partial C} = 
\begin{bmatrix}
\frac{\partial L}{\partial c_{11}} & \frac{\partial L}{\partial c_{12}} & \frac{\partial L}{\partial c_{13}} \\
\frac{\partial L}{\partial c_{21}} & \frac{\partial L}{\partial c_{22}} & \frac{\partial L}{\partial c_{23}} \\
\frac{\partial L}{\partial c_{31}} & \frac{\partial L}{\partial c_{32}} & \frac{\partial L}{\partial c_{33}}
\end{bmatrix} = 
\begin{bmatrix}
0 & 1 & 2 \\
3 & 4 & 5 \\
6 & 7 & 8 
\end{bmatrix}
$$

，那么loss或者上一层的输出对A的第一行第一列元素的偏导数用链式法则计算为：

$$
\frac{\partial L}{\partial a_{11}} = \frac{\partial L}{\partial C} \frac{\partial C}{\partial a_{11}} =
\sum
\begin{bmatrix}
\frac{\partial L}{\partial c_{11}} & \frac{\partial L}{\partial c_{12}} & \frac{\partial L}{\partial c_{13}} \\
\frac{\partial L}{\partial c_{21}} & \frac{\partial L}{\partial c_{22}} & \frac{\partial L}{\partial c_{23}} \\
\frac{\partial L}{\partial c_{31}} & \frac{\partial L}{\partial c_{32}} & \frac{\partial L}{\partial c_{33}}
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{\partial c_{11}}{\partial a_{11}} & \frac{\partial c_{12}}{\partial a_{11}} & \frac{\partial c_{13}}{\partial a_{11}} \\
\frac{\partial c_{21}}{\partial a_{11}} & \frac{\partial c_{22}}{\partial a_{11}} & \frac{\partial c_{23}}{\partial a_{11}} \\
\frac{\partial c_{31}}{\partial a_{11}} & \frac{\partial c_{32}}{\partial a_{11}} & \frac{\partial c_{33}}{\partial a_{11}}
\end{bmatrix} = 
\sum
\begin{bmatrix}
0 & 1 & 2 \\
3 & 4 & 5 \\
6 & 7 & 8 
\end{bmatrix}
\cdot
\begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix} = 0 \times b_{11} + 1 \times b_{12} + 2 \times b_{13} = 20
$$
，如此计算L对A的每一个元素的偏导数即可得到上一层输出对A的梯度$\frac{\partial L}{\partial A}$。
