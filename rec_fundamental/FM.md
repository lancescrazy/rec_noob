# FM

## 1. FM模型的引入

### 1.1 逻辑回归模型及其缺点

FM更应该理解为是一种思路，应用较少。做推荐CTR预估时最简单的技术就是将特征做线性组合（逻辑回归LR），传入sigmoid中，得到概率值(线性模型)。sigmoid是单调函数，不会改变里面的线性模型的CTR预测顺序，因此回归模型效果会差，即LR的缺点：

> - 是一个线性模型
> - 每个对最终输出结果独立，需要手动特征交叉 x_i * x_j 

### 1.2 二阶交叉项的考虑及改进

由于LR模型的缺陷（手动特征交叉）干脆考虑所有的二阶交叉项

原目标函数为：
$$
y = w_0+\sum_{i=1}^nw_ix_i
$$
改进目标函数为：
$$
 y = w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^{n-1}\sum_{i+1}^nw_{ij}x_ix_j 
$$
**存在一个问题**：
$$
只有当x_i和x_j都不为0时，这个二阶交叉项才会生效
$$
（这个特征交叉项本质是和多项式核SVM等价的）

FM就是为了解决这个问题：

使用优化函数：
$$
y = w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^{n}\sum_{i+1}^n\lt v_i,v_j\gt x_ix_j
$$
变动很明显，这里面就有深度学习的味道了，实质就是：
$$
给每个x_i计算一个Embedding，然后将两个向量间的Embedding做内积，得到之前所谓的w_{ij}
$$
好处是模型泛化能力强，即使两个特征之前从未在训练集中从未<u>**同时**</u>出现，也不至于训练不出w_ij，只需要x_i和其他的x_k同时出现过就可以计算出x_i的Embedding

## 2. FM公式的理解

模型前半部分是普通的LR线性组合，后半部分的交叉项：特征组合，模型表现力更强
$$
对于有n个特征的模型，特征组合的参数数量共有 1+2+3+\cdots + n - 1 = \frac{n(n-1)}{2}个
$$
且任意两参数独立，维度增加
$$
定理：任意一个实对称矩阵（正定矩阵）W都存在一个矩阵V，使得W = V \cdot V^T成立
$$

$$
类似的，所有二次项参数\omega_{ii}可以组成一个对称阵W（为方便，设置对角元素为正实数）
$$

$$
则这个矩阵可以分解为W = V^TV，V的第j列(v_{j})便是第j维特征x_j的隐向量
$$

$$
\hat{y}(X) = \omega_{0}+\sum_{i=1}^{n}{\omega_{i}x_{i}}+\sum_{i=1}^{n-1}{\sum_{j=i+1}^{n} \color{pink}{<v_{i},v_{j}>x_{i}x_{j}}}
$$

$$
需要估计的参数有\omega_{0} ∈ R，\omega_{i}∈ R，V∈ R，< \cdot, \cdot>是长度为k的两个向量的点乘
$$

$$
<v_{i},v_{j}> = \sum_{f=1}^{k}{v_{i,f}\cdot v_{j,f}} 
$$

> $$
> \begin{cases}
> 	\omega_{0}为全局偏置；
>     \\
>     \omega_{i}是模型第i个变量的权重;
>     \\
>     \omega_{ij} = < v_{i}, v_{j}>特征i和j的交叉权重;
>     \\
>     v_{i} 是第i维特征的隐向量;
>     \\
>     <\cdot, \cdot>代表向量点积;
>     \\
>     k(k<<n)为隐向量的长度，包含 k 个描述特征的因子。
> \end{cases}
> $$

二次项的参数数量为 kn 个，远少于多项式模型的参数数量，参数因子化使得x_h, x_i的参数和x_i, x_j的参数不再是相互独立的，因此可以在样本稀疏的情况下相对合理地估计FM的二次项参数。

具体的说：
$$
x_{h}x_{i} 和 x_{i}x_{j}的系数分别为 \lt v_{h},v_{i}\gt 和 \lt v_{i},v_{j}\gt ，它们之间有共同项v_{i} 
$$

$$
那么所有包含“ x_{i} 的非零组合特征”（存在某个 j \ne i ，使得 x_{i}x_{j}\neq 0）的样本都可以用来学习隐向量v_{i}
$$

这很大程度上，避免了数据稀疏性造成的影响，而在多项式模型中，w_hi和w_ij是相互独立的

显而易见：FM的公式是一个通用的拟合方程，可以采用不同的损失函数来解决回归、分类问题，如
MSE（mean square error）loss function来求解回归问题，也可以使用Hinge/Cross-Entropy loss来解决分类问题。

在进行二分类时，FM的输出需要使用sigmoid函数进行交换，原理见LR。

FM在线性时间内完成对新样本的预测

证明
$$
\begin{align} \sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{<v_i,v_j>x_ix_j}} &= \frac{1}{2}\sum_{i=1}^{n}{\sum_{j=1}^{n}{<v_i,v_j>x_ix_j}} - \frac{1}{2} {\sum_{i=1}^{n}{<v_i,v_i>x_ix_i}} \\ &= \frac{1}{2} \left( \sum_{i=1}^{n}{\sum_{j=1}^{n}{\sum_{f=1}^{k}{v_{i,f}v_{j,f}x_ix_j}}} - \sum_{i=1}^{n}{\sum_{f=1}^{k}{v_{i,f}v_{i,f}x_ix_i}} \right) \\ &= \frac{1}{2}\sum_{f=1}^{k}{\left[ \left( \sum_{i=1}^{n}{v_{i,f}x_i} \right) \cdot \left( \sum_{j=1}^{n}{v_{j,f}x_j} \right) - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2} \right]} \\ &= \frac{1}{2}\sum_{f=1}^{k}{\left[ \left( \sum_{i=1}^{n}{v_{i,f}x_i} \right)^2 - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2} \right]} \end{align}
$$

$$
\begin{cases}
    v_{i,f} 是一个具体的值
    \\
    第1个等号：对称矩阵 W 对角线上半部分；
    \\
    第2个等号：把向量内积 v_{i},v_{j} 展开成累加和的形式；
    \\
    第3个等号：提出公共部分；
    \\
    第4个等号： i 和j相当于是一样的，表示成平方过程。
\end{cases}
$$

## 4. 代码实践

### 4.1 掉包安装

[github官方仓库（安装与使用）](https://github.com/coreylynch/pyFM)

直接pip install安装

```
pip install git+https://github.com/coreylynch/pyFM
```

### 4.2 掉包测试

#### 4.2.1 导包

```python

```



