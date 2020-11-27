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

## 五、代码实战

1. 使用tensorflow中已经封装好的wide&deep模型，这一部分主要熟悉模型训练的整体结构
2. tensorflow中的keras实现wide&deep，尽可能看到模型内部的细节，并将其实现

### 1. Tensorflow内置的WideDeepModel

全局实现：

```python
tf.keras.experimental.WideDeepModel(
	linear_model, dnn_model, activation=None, **kwargs
)
```

这一步就是将 linear_model 与 dnn_model 拼接在一起，对应于 Wide-Deep FM 中的最后一步

如：做一个简单的实现

```python
linear_model = LinearModel()
dnn_model = keras.Sequential([keras.layers.Dense(units=64),
                             keras.layers.Dense(units=1)])
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd','adam'], 'mse', ['mse'])
# define dnn_inputs and linear_inputs as separate numpy arrays or 
# a single numpy array if dnn_inputs is same as linear_inputs.
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
# or define a single 'tf.data.Dataset' that contains a single tensor or 
# separate tensors for dnn_inputs and linear_inputs
dataset = tf.data.Dataset.from_tensors(([linear_inputs, dnn_inputs], y))
combined_model.fit(dataset, epochs)
```

第一步是直接调用 keras.experimental 中的 linear_model ，第二步简单实现了一个全连接神经网络，第三步使用 WideDeepModel 将前两步产生的两个 model 拼接在一起，形成最终的 combined_model，接着就是常规的 compile 和 fit 了。

除此之外线性模型与DNN模型在联合训练之前均可进行分别训练：

```python
linear_model = LinearModel()
linear_model.compile('adagrad', 'mse')
linear_model.fit(linear_inputs, y, epochs)
dnn_model = keras.Sequential([keras.layers.Dense(units=1)])
dnn_model.compile('rmsprop', 'mse')
dnn_model.fit(dnn_inputs, y, epoches)
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
```

前三行代码训练一个线性模型，中间三行训练一个 DNN 模型，最后三行将两个模型联合训练

**Tensorflow 实现 wide&deep 模型**

这一部分对原始特征进行转换，以及 deep 特征和 wide 特征的选择，特征的交叉一系列特征操作，模型也分成了 wide 部分和 deep 部分，这里为了简单实现，使用了同一个优化器优化两部分

## 六、课后思考

Wide&Deep 模型仍然存在哪些不足，针对这些不足，工程师们有哪些改进

### 1 模型的训练效率提升

> 稀疏矩阵相乘
>
> 数据输入时，用 Tensorflow Data API 代替最初的 feed_dict
>
> 用 TFRecords 文件格式代替最初的 csv 来存储训练参数，并且实现在 hadoop 上并发成 TFRecords 数据文件
>
> 多进程并行：多个读进程将 HDFS 上的 TFRecords 下载到本地，一个进程负责训练
>
> 多GPU同时进行并行训练

### 2 新版本 TF

新版本 TF 自带了 DNNLinearCombinedClassifier 实现了 Wide&Deep 模型

> 几行代码即可
>
> 继承自 Estimator，基类功能：
>
> - 定时保存模型
> - 重启后自动加载模型继续训练
> - 自动保存 metric 供模型可视化、分布式训练
>
> 阅读源码：掌握TensorFlow高级API方法，摆脱Estimator限制，用Tensorflow底层API实现更复杂模型

### 3 概览

第一部分：

> 开场白：
>
> “记忆与扩展”，“类别特征”，“特征交叉”
>
> Wide&Deep, FM/FFM/DeepFM, Deep&Cross Network, Deep Interest Network

第二部分：

> Feature Column:
>
> 特征处理方法：特征工程是 Wide&Deep 的精华，特别是对 Categorical 特征的处理

第三部分：

> DNNLinearCombinedClassifier

最后：删除业务细节开源代码

### 4 Wide & Deep 的三个关键词

#### 4.1 记忆与扩展

记忆 Memorization， 扩展 Generalization

Exploitation & Exploration，著名的 EE 问题

Wide 侧重于记忆？能记住什么，将什么样的特征输入Wide

> 历史数据中常见，高频的模式，是推荐系统中，红海
>
> 重点学习模式之间的权重，做模式的筛选
>
> 由不能发现新模式：根据人工经验，业务背景，将我们认为有价值的、显而易见的特征及特征组合，喂入Wide

也要能从历史数据中发现低频、长尾的模式，发现用户兴趣的蓝海，即具备良好的扩展能力

Deep 侧

> 通过 embedding ＋ 深层交互，能够学到国籍、节日、食品各种 tag 的最优向量表示
>
> 推荐引擎给<中国人、感恩节、火鸡>这种新组合，可能会打一个不低的分数
>
> 即通过 Embedding 将 tag 向量化，变 tag 的精确匹配，为 tag 向量的模糊查找，使得自己具备了良好的“扩展能力”

#### 4.2 类别特征

深度学习热潮：发源于CNN在图像识别上取得的巨大成功，后来才扩展到推荐、搜索等领域

实际上两者间有很大的不同，其中重要的不同，就是图像都是稠密特征，而推荐搜索中，都是稀疏的类别、ID类特征，Google在《Ad Click Prediction: a View from the Trenches》：**因为稀疏/稠密的区别，CNN中效果良好的Dropout技术，运用到CTR预估，推荐领域反而会恶化性能**

相比于实数型特征，**稀疏的类别、ID类特征，才是推荐、搜索领域的公民**，被更广泛研究，即使有一些实数值特征，如：历史曝光次数、点击次数、CTR等，也往往通过 bucket 的方式变成 categorical 特征，才喂进模型

​       推荐、搜索喜欢稀疏的类别/ID特征，主要有三个原因    

> LR，DNN 在底层还是一个线性模型，但在生活中，**标签 y 和特征 x 之间较少存在线性关系，往往是分段的**，以“点击率 ～ 历史曝光次数”间关系为例，曝光过1~2次，是正相关，再曝光1~2次，用户由于好奇，没准就点击了；但是若曝光了8~9次，用户失去新鲜感，越多曝光，越不能再点，呈现出负相关，因此，categorical 特征相比于 numeric 特征，更符合现实场景
>
> 推荐、搜索一般都是基于用户、商品的标签画像系统，而标签天生就是Categorical的
>
> 稀疏的类别/ID类特征，可以稀疏的**存储，传输，运算，提升运算速率**

**但是**，稀疏的categorical/ID类特征，也有着**单个特征表达能力弱、特征组合爆炸、分布不均匀导致受训程度不均匀**，为解决这些问题：

> FTRL这样的算法，充分利用输入的稀疏性在线更新模型，训练处的模型也是稀疏的，便于快速预测
>
> Parameter Server分布式系统，充分利用特征的稀疏性，不必再各机器之间同步全部模型，而让每台机器按需同步自己所需要的部分模型权重，按需要上传这一部分权重的梯度，提升分布式计算的效率
>
> TensorFlow Feature Column类，除了一个 numeric_column 是处理实数特征，其他都是围绕处理 Categorical特征的，封装了常见的分桶、交叉、哈希

#### 4.3 特征交叉

特征交叉，增强 categorical 特征的表达能力。围绕着如何做特征交叉

假设Categorical按照one-hot-encoding展开，共n个特征，很稀疏

若在LR中加入特征交叉，只考虑二项式交叉，要学习 1(bias) + n(一次项) + 1/2*n(n-1)(二次项） = 1+n+0.5n(n-1)，计算量大容易过拟合，大量的交叉项x_ix_j 由于在训练数据中稀少甚至没有，所以，为此发明了Factorization Machine（FM）算法，第 i 维度，对应一个k维隐向量v_i，交叉项的权重有两个隐向量的内积表示<v_i, v_j>
$$
y(x) = w_0 + \sum_{i=1}^{n}w_ix_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j
$$
FM的优势：

- 将需要优化的权重，由1+n+0.5n(n-1)减小到1+n+n*k，而k<<n，即减少了计算量，也降低了过拟合的风险
- 原公式只有x_ix_j都不为0，w_ij才有训练的机会。所有FM公式中，所有x_i 不等于0的样本都可以训练v_i，而所有的x_j不等于0的样本都可以训练v_j，权重得到训练的机会大大增加

FM 一般只限于二次特征交叉，而深度神经网络 DNN 先将 Categorical / id 特征通过 Embedding 映射成稠密向量，再喂入 DNN ，让 DNN 自动学习到这些特征之间的深层交叉

Wide & Deep:

> - wide 侧 LR，一般根据人工先验知识，将一些简单明显的特征交叉，喂入 wide 侧，让 wide 侧能够记住这些规则
>
> - Deep 就是 DNN ，通过 Embedding 的方式将 Categorical / id 特征映射成稠密向量，让 DNN 学习到这些特征之间的深层交叉，以增强扩展能力

DeepFM在wide侧用一个FM模型替换了LR，能够自动学习到所有二次项的系数

> 关键在Deep侧与wide侧共享一个Embedding矩阵来映射categorical/id特征到稠密向量
>
> ​                      
>
> Deep将Embedding结果喂入DNN，来学习深层交互的权重，着重扩展
>
> Wide将Embedding结果喂入FM，来学习二次交互的权重，着重记忆

但是DNN的深层交互是隐式的，不知道学习的是那些特征的几阶交互，google的Deep&cross network允许显式指定交叉阶次，并高效学习

![tag](https://upload-images.jianshu.io/upload_images/25239821-6206b7345b04359f.jpg?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

Feature Column是特征预处理器，他和输入数据间关系如下

![特征预处理器](https://upload-images.jianshu.io/upload_images/25239821-dc37b12321b9b93a.jpg?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

def input_fn()

> "SepalLength": [...],
>
> features - labels

tf.feature_column

> numeric_column("SepalLength")
>
> how to bridge input to model
>
> match feature names from input_fn 

tf.estimator.DNNClassifier

> classifier = DNNClassifier(feature_columns=feature_columns,
>
> ​											hidden_units=[10, 10],
>
> ​											n_classed=3,
>
> ​											model_dir=PATH)

- Feature Column本身不存储数据，只是封装一些预处理逻辑，比如输入的字符串（tag），把这些字符串根据字典映射成id，再根据id映射成Embedding vector这些预处理逻辑有不同的feature column

- Input_fn 返回的data_set可以看成{feature_name: feature tensor}的 dict ，而每个 feature column 定义时需要指定一个名字，feature column 与 input 通过这个名字联系在一起

关系图如下：其中只有一个numeric_column是纯粹处理数值特征，其余都与处理Categorical特征有关

![](https://upload-images.jianshu.io/upload_images/25239821-7c670400a9877356.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/720/format/webp)

重要说明：

> Feature column的实现在：tensorflow/python/feature_column/feature_column.py
>
> 删除了一些异常处理，assert，检查type，logging等辅助代码

### 5 基类 FeatureColumn，DenseColumn，CategoricalColumn

___FeatureColumn 是所有 feature column 的基类_

比较重要的是，__transform_feature(self, inputs) 虚函数，各子类主要的预处理逻辑都是通过重载这个函数来实现的

基类__DenseColumn是所有numeric/dense feature column的基类，比较重要的是：get_dense_tensor(self, inputs, ..)虚函数

> - inputs可以理解为从input_fn返回的dict of input tensor的wrapper。inputs一般是__LazyBuilder类型的，除了实现按列名查找input tensor的



















