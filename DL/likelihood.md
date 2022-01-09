# Cross Entropy and Log Likelihood

- **Probability**: attaches to possible results. 是指在固定参数的情况下，事件的概率，必须是0-1，事件互斥且和为1. 我们常见的泊松分布、二项分布、正态分布的概率密度图描述的就是这个。
- **Likelihood**: attaches to hypotheses. 是指固定的结果，我们的参数的概率，和不必为1，不必互斥，所以只有两个likelihood的ratio（$\frac{L(\theta_1)}{L(\theta_2)} > 1$）是有意义的。

## Single observation
Given input $x$, the vector $\hat{y}=h(x)$ is of length $M$ (the number of label) and can be interpreted as the probability of each of $M$ possible outcomes occurring, according to the model represented by the neural network.

**Negative log-likelihood is equals to cross entropy when both as loss function in the neural work.**

### Log Likelihood
Choose the model that maximizes the likelihood of the observed data. I.e., we want to find the value of the parameter $\theta$ that maximizes the likelihood of the data by using negative log-likelihood as a cost function.

If we make a single observation, and we observe outcome $j$, then the **likelihood** is simply $y_j$.

If we represent the actual observation as a vector $y$ with one-hot encoding, then the **likelihood of the same single observation** can be represented as $\prod_{j=1}^{M}{\hat{y}_{j}^{y_j}}$. (Since each term in the product will be equal to 1 except that corresponding to the observed value).

The negative log likelihood is then
$$
NLL = - \sum^{M}_{j=1}{y_j \log \hat{y}_j} \tag{1}
$$
where the vector $\hat{y}$ represents a discrete probability distribution over the possible values of the observation (according to our model). The vector $y$ can also be interpreted as a probability distribution over the same space, that just happens to give all of its probability mass to a single outcome (i.e., the one that happened). We might call it the empirical distribution.

Under this interpretation, the expression for the negative log-likelihood above is also equal to a quantity known as the cross entropy.


### Cross Entropy
If a discrete random variable $X$ has the probability mass function $f(x)$, then the *entropy* of $X$ is
$$
H(X) = \sum_x f(x) \log \frac{1}{f(x)} = - \sum_x f(x) \log f(x) \tag{2}
$$
It is the expected number of bits needed to communicate  the value taken by $X$ if we use the optimal coding scheme for the distribution.

The expression for **cross entropy** is
$$
H(g, f) = \sum_x g(x) \log \frac{1}{f(x)} = - \sum_x g(x) \log f(x) \tag{3}
$$
where $f$ is the distribution we used to generate our coding scheme (i.e., the distribution that we think the data follows), and $g$ is the true distribution.
Note that **($H(g, f) \neq H(f, g)$)**.

In the neural network, the cross entropy as cost function is,
$$
-\sum^{M}_{j=1} y_i \log \hat{y}_j
\tag{4}
$$
where $\hat{y}$ represents the distribution that the model believes the data follows, and $y$ is the actual data, and so is the true distribution.


### K-L Divergence
The Kullback-Leibler (K-L) divergence is the number of additional bits, on average, needed to encode an outcome distributed as $g(x)$ if we use the optimal encoding for $f(x)$. The K-L divergence is often described as a meature of the distance between distributions.
The K-L divergence is
$$
D_{KL}(g||f) = H(g, f) - H(g) = -(\sum_x g(x) \log f(x) - \sum_x g(x) \log g(x))
\tag{5}
$$

## Multiple observations
The joint likelihood is
$$
\prod^{N}_{i=1} \prod^{M}_{j=1} \hat{y_j^{(i)}}^{y_j^{(i)}}
\tag{6}
$$
and the negative log likelihood is
$$
-\sum^{N}_{i=1} \sum^{M}_{j=1} {y_j^{(i)}} \log \hat{y_j^{(i)}}
\tag{7}
$$
where N is the number of independently sampled examples from a training data set, $y_j^{(i)}$ is the outcome of the $i$th example, and $\hat{y_j^{(i)}}$ is the likelihood of that outcome according to the model.

The per-example negative log likelihood can indeed be interpreted as cross entropy. However, the negtive log likelihood of a batch data (which is just the sum of the negative log likelihoods of the individual examples) seems to be not a cross entropy, but a sum of cross entropy each based on a different model distribution (since the model is conditional on a different $x^{(i)}$ for each $i$).
A true opinion about cross entropy over multiple observations, is the cross entropy between the distribution over the vector of outcomes for the batch data and the probability distribution over the vector of outcomes given by our model, i.e., $p(y|X,\theta)$, with each distribution being conditional on the batch of observed values $X$.


## References

[1] [Cross entropy and log likelihood](http://www.awebb.info/blog/cross_entropy)
[2] [Probability和Likelihood的区别](https://www.cnblogs.com/leezx/p/9225881.html)
[3] [Bayes for Beginners: Probability and Likelihood](https://www.psychologicalscience.org/observer/bayes-for-beginners-probability-and-likelihood)
[4] [Likelihood Ratio](https://onlinelibrary.wiley.com/doi/pdf/10.1002/0470011815.b2a15073)