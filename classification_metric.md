### 基本概念

#### 准确率（accuracy）
$$
ACC = \frac{TP + FN}{TP + FP + TN + FN}
$$

#### 精确率（precision）
查准率，正确的预测样本占所有预测为正的样本的比例。

$$
precision = \frac{TP}{TP + FP}
$$

#### 召回率（recall）= TPR（true positive rate）= 敏感度（sensitivity）
查全率，预测为正的正样本占所有正样本的比例。

$$
recall = TPR = sensitivity = \frac{TP}{TP + FN}
$$

#### 特异度（specificity）= TNR(true negative rate)
预测为负的样本占所有负样本的比例。

$$
specificity = TNR = 1 - FPR = \frac{TN}{TN + FP} \\
FPR = \frac{FP}{TN + FP}
$$

其中，FPR(false positive rate)为预测为正的负样本占所有负样本的比例。


#### F1 Score
precision和recall的平衡指标。

$$
f1 = \frac{2}{\frac{1}{recall} + \frac{1}{precision}} = \frac{2 \times precision \times recall}{precision + recall}
$$
