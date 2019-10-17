# Confidence Calibration

## 1. Introduction
- Calibrated Confidence: the probability associated with the predicted class label should reflect its ground truth correctness likelihood.


## 2. Definitions

### Basis

**Input and label**: $X \in \chi$ and $Y \in \{1, 2, ..., K\}$, both follow a ground truth joint distribution $\pi(X, Y) = \pi(Y|X)\pi(X)$.

**Neural network**: 
$$
h(X)=(\hat{Y},\hat{P})	\tag{2.1}
$$
, where $\hat{Y}$ is class prediction, and $\hat{P}$ is its associated confidence(probability of correctness).

**Perfect Calibration**: 
$$
\mathbb{P}(\hat{Y}=Y | \hat{P}=p) = p, \forall p \in [0,1] \tag{2.2}
$$
, where $\hat{P}$ is a continuous random variable.

**Reliability Diagrams**: group predictions into $M$ interval bins (each of size $\frac{1}{M}$). And $B_m$ is the set of samples whose prediction confidence falls into the interval $I_m =[a_m, a_{m+1})=[\frac{m-1}{M}, \frac{m}{M})$. The accuracy and the average confidence of $B_m$  are respectively 
$$
\begin{align}
acc(B_m) =& \frac{1}{|B_m|}\sum_{i \in B_m}{1(\hat{y_i} = y_i)} \tag{2.3} \\
conf(B_m) =& \frac{1}{|B_m|}\sum_{i \in B_m}{\hat{p_i}} \tag{2.4}
\end{align}
$$
The $acc(B_m)$ and $conf(B_m)$ approximate the left-side and right-side of Eq(2.2) respectively for bin $B_m$. Therefore, a perfectly calibrated model will have $acc(B_m) = conf(B_m)$ for all $m \in \{1, \ldots, M \}$

### Items

**Expected Calibration Error (ECE)**: a miscalibration notion and the difference in expectation between accuracy and confidence,  
$$
ECE = \mathbb{E}[|\mathbb{P}(\hat{Y}=Y|\hat{P}=p)-p|] \tag{2.5}
$$
Or approximates Eq(2.5) by partitioning predictions into $M$ and taking a weighted average of bin's accuracy/confidence difference,
$$
ECE = \sum^{M}_{m=1}{\frac{|B_m|}{n}|acc(B_m)-conf(B_m)|} \tag{2.6}
$$
where $n$ is the number of samples. The difference between $acc$ and $conf$ for a given bin represents the calibration gap.

**Maximum Calibration Error (MCE)**:
$$
\begin{align}
MCE = \max_{p \in [0,1]}|\mathbb{P}(\hat{Y}=Y|\hat{P}=p)-p| \tag{2.7} \\
MCE = \max_{m \in \{1,\ldots,M\}}|acc(B_m) - conf(B_m)| \tag{2.8}
\end{align}
$$
For perfectly calibrated classifiers, MCE and ECE both equal 0.

**Negative Log Likelihood (NLL)**: a standard measure of a probabilistic model's quality and be referred to as the cross entropy loss in the context of deep learning.
$$
NLL = -\sum^{n}_{i=1}{log(\hat{\pi}(y_i|x_i))} \tag{2.9}
$$
where $\hat{\pi}(Y|X)$ is a probabilistic model and n is the number of samples. And NLL is minimized if and only if $\hat{\pi}(Y|X)$ recovers the ground truth conditional distribution $\pi(Y|X)$.

## 2. The factors affecting calibration

**Model Capacity**: increased model capacity negatively affect model calibration.

**Batch Normalization**: models trained with Batch Normalization tend to be more miscalibrated.

**Weight Decay**: models trained with less weight decay have a negatively impact on calibration.

**NLL**: Between NLL and accuracy have a disconnect because of neural networks can overfit to NLL without overfitting to the 0/1 loss. *The observed disconnect between NLL and 0/1 loss suggests that these high capacity models are not necessarily immune from overfitting, but rather, overfitting manifests in probabilistic error rather than classification error*.

**All above factors will impact classification accuracy**.


## 3. Calibration Methods

All methods are post-processing steps that produce (calibrated) probabilities. And the parameters of calibration method are optimized over the validation set, while the parameters of model are fixed.

### Calibrating Binary Models

Label: $y=\{0,1\}$. Given a sample $x_i$, we have access to $\hat{p}$ - the model's predicted probability of $y_i=1$, as well as $z_i \in \mathbb{R}$ - which is the model's non-probabilistic output or logit. The $\hat{p}$ is derived from $z_i$ using a sigmoid function $\sigma$ (i.e. $\hat{p_i}=\sigma(z_i)$). 
Our goal is to produce a calibrated probability $\hat{q_i}$ based on $y_i, \hat{p_i}$ and $z_i$.

**Histogram binning**
$\hat{q_i}=\theta_m$ if $\hat{p_i}$ is assigned to bin $B_m$. and the bin boundaries are either chosen to be equal length intervals or to equalize the number of samples in each bin.
The predictions $\theta_i$ are chosen to minimize the bin-wise squared loss:
$$
\min_{\theta_1,\ldots,\theta_M}\sum^{M}_{m=1}\sum^{n}_{i=1}{1(a_m \leq \hat{p_i} < a_{m+1})(\theta_m - y_i)^2} \tag{3.1}
$$

**Isotonic regression (strict generation of histogram binning)**
The bin boundaries and bin predictions are jointly optimized,
$$
\min_{\theta_1,\ldots,\theta_M}
\min_{a_1,\ldots,a_{M+1}}
\sum^{M}_{m=1}\sum^{n}_{i=1}
{1(a_m \leq \hat{p_i} < a_{m+1})(\theta_m - y_i)^2} 
\tag{3.2} \\
subject \  to \  0 \leq a_1 \leq a_2 \leq \ldots a_{M+1}=1, \\
\theta_1 \leq \theta_2 \leq \ldots \theta_M.
$$

**Bayesian Binning into Quantiles (BBQ)**
Extension of histogram binning using Bayesian model averaging.

**Platt Scaling**

$\hat{q_i} = \sigma(az_i + b)$ as the calibrated probability.
Parameters $a$ and $b$ can be optimized using the NLL loss over the validation set.


### Calibrating Multiclass Models
The label is $y=\{0,1,\ldots,K\}(K > 2)$, $\hat{y_i}=argmax_{k} z_i^{(k)}$, and $\hat{p_i}$ is typically derived using softmax function $\sigma_{SM}$,
$$
\sigma_{SM}{(z_i)^{(k)}} = \frac{exp(z_i^{(k)})}{\sum^K_{j=1}exp(z_i^{(j)})},
\hat{p_i} = \max_{k}{\sigma_{SM}(z_i)^{(k)}} \tag{3.3}
$$
The goal is to produce a calibrated confidence $\hat{q_i}$ and (possibly new) class prediction $\hat{y_i^{'}}$ based on $y_i$, $\hat{y_i}$, $\hat{p_i}$ and $z_i$.

**Extension of binning methods**
Treating the multiclass problem as $K$ one-versus-all problems.

**Matrix and vector scaling**
$$
\begin{align}
\hat{q_i} &= \max_{k}{\sigma_{SM}{(Wz_i + b)^{(k)}}}, \\
\hat{y_i^{'}} &= \mathop{\arg\max}_k(Wz_i + b)^{(k)}.
\tag{3.4}
\end{align}
$$

**Temperature Scaling (most useful method)**
The simplest extension of Platt scaling. The new confidence prediction is
$$
\hat{q_i} = \max_{k}{\sigma_{SM}(\frac{z_i}{T}})^{(k)} \tag{3.5}
$$
, where $T$ is called the temperature, and it "softens" the softmax (i.e. raises the output entropy) with $T>1$. $T$ is optimized with respect to NLL on validation set.
*Temperature scaling does not affect the model's accuracy. The model is equivalent to maximizing the entropy of the output probability distribution subject to certain constraints on the logits.*

### References: 
[1] [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
