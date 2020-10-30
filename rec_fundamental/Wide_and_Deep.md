# Wide & Deep

## 一、点击率预估简介

### 1. 点击率是用来解决什么问题

对每次广告点击情况做出预测，可以输出点击或者不点击，也可以输出该次点击的概率，后者称为pClick

### 2. 点击率预估模型需要做什么

通过点击率评估的基本概念，发现点击率评估问题就是一个二分类的问题，机器学习中，可以使用逻辑回归作为模型输出

### 3. 点击预估与推荐算法的不同

广告点击率得到用户对对应广告的点击率，结合广告的出价用于排序

推荐算法要得到一个最优排序，TopN推荐的问题，可以利用广告的点击率来排序，作为广告的推荐

## 二、为什么不用FM呢

FM的缺点在于：当query-item矩阵是稀疏并且是high-rank时（user有特殊爱好，item小众），很难有效率的学习出低维度的表示

这种情况下，大部分的query-item都没有什么关系。

但是dense Embedding会导致所有的query预测值非0，导致推荐过度泛化，推荐不相关物品

而简单的linear model可以通过 cross-product transformation来记住这些exception rules，

## 三、Wide & Deep模型的“记忆能力”与“泛化能力”

### 1. 基本概念

推荐系统常见概念：

> Memorization：用户与商品的交互信息矩阵学习规则
>
> Generalization：泛化规则

FM是很典型的Generalization，根据互信息学到比较短的矩阵V，其中v_i中储存着每个用户特征的压缩表示（Embedding），协同过滤和SVD都是靠记住用户之前与那些物品发生了交互而腿短的推荐结果。Wide&Deep是结合了这两种推荐结果做出的推荐，该模型效果更好

Memorization更保守，推荐用户之前有过行为的items，只需要线性模型就可以实现

Generalization更趋向于提高推荐系统的diversity，需要使用DNN实现

![示意图](https://i.pinimg.com/originals/e4/49/58/e44958c38b21f473c55ac9dd39b081fe.png)

难点在于：如何根据自己的场景取选择特征放在Wide部分，哪些特征放在Deep部分，

### 2. 如何理解

如何理解Wide部分有利于增强模型的记忆能力，Deep部分有利于增强模型的泛化能力

#### 2.1 广义现行模型Wide部分

输入特征由两部分组成：原始的部分特征，以及原始特征的交互特征，cross product transformation

交互特征的定义：一种特征组合的定义，只有两个特征同时为1，新特征才为1
$$
\phi_{k}(x)=\prod_{i=1}^d x_i^{c_{ki}}, c_{ki}\in {0,1}
$$
`AND(user_installed_app=QQ, impression_app=WeChat)，当特征user_installed_app=QQ,和特征impression_app=WeChat取值都为1的时候，组合特征AND(user_installed_app=QQ, impression_app=WeChat)的取值才为1，否则为0。`

对于wide部分训练时候使用的优化器是带L_1正则的FTRL算法（Follow-the-regularized-leader）而L1 FTLR非常注重模型稀疏性质，W&D模型采用L1 FTRL是想让Wide部分更稀疏，即Wide部分的大部分参数都为0，从而压缩模型权重及特征向量的维度。**Wide部分模型训练完后留下来的特征都非常重要，模型的记忆能力理解为发现“直接的”，“暴力的”，“显然的“，关联规则的能力**，如Google W & D期望wide部分发现这样的规则：用户安装了应用A，此时曝光应用B，用户安装应用B的概率大

#### 2.2 DNN模型Deep部分

输入特征主要分为两大类：数值特征（直接输入），类别特征（Embedding后输入到DNN），Deep数学部分形式如下：
$$
a^{(l+1)} = f(W^{l}a^{(l)} + b^{l})
$$
**DNN模型随着层数增加，中间特征越抽象，提高模型的泛化能力**，使用优化器AdaGrad，提高解的精度

#### 2.3 Wide部分与Deep部分的结合

结合起来训练，输出重新使用一个逻辑回归模型，做最终的预测，输出概率值。联合训练的数学形式如下：
$$
P(Y=1|x)=\delta(w_{wide}^T[x,\phi(x)] + w_{deep}^T a^{(lf)} + b) 
$$

## 四、操作流程

> Retrieval：利用机器学习和人为定义的规则，返回最匹配当前Query的items集合，这个集合就是最终的推荐列表候选集
>
> Ranking：
>
> - 收集更细致的用户特征，如：
>
>   - User features（年龄，性别，语言，民族）
>
>   - Contextual features（上下文特征：设备，时间等）
>
>   - Wide组件是FTRL + L1正则学习
>
>     Deep是AdaGrad学习
>
>     训练完推荐TopN
>
>     Wide&Deep需要深入理解业务，确定wide部分使用哪部分特征，deep部分使用那些特征，wide部分的交叉验证如何去选择Impression features（展示特征：app age，app历史统计信息等）
>
> - 将特征传入Wide和Deep一起训练：根据最终的loss计算出gradient，反向传播到Wide和Deep两部分中，分别训练自己的参数（wide组件主要填补deep组件的不足，是较少的cross product feature transformation而不是full size model
>
>   - 训练方法：mini batch stochastic optimization
>   - Wide组件是FTRL + L1正则学习
>   - Deep是AdaGrad学习
>
> - 训练完推荐TopN

Wide&Deep需要深入理解业务，确定wide部分使用哪部分特征，deep部分使用那些特征，wide部分的交叉验证如何去选择























