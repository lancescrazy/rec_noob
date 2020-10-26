import numpy as np
import math
class SVD(object):
    def __init__(self, rating_data, F=5, alpha=0.1, lmbda=0.1, max_iter=100):
        self.F = F  # 隐向量的长度
        self.P = dict()  # 用户矩阵P 大小是[users_num, F]
        self.Q = dict()  # 物品矩阵Q 大小是[items_num, F]
        self.bu = dict()  # 用户偏差系数
        self.bi = dict()  # 物品偏差系数
        self.mu = 0.0  # 全局偏差系数
        self.alpha = alpha  # 学习率
        self.lmbda = lmbda  # 正则项系数
        self.max_iter = max_iter  # 最大迭代次数
        self.rating_data = rating_data  #　评分矩阵

        # 初始化矩阵 P 和 Q，方法很多，一般用随机数填充，随机数大小根据经验需要和 1/sqrt(F) 成正比
        cnt = 0  # 统计总的打分数，初始化mu用
        for user, items in self.rating_data.items():
            self.P[user] = [np.random.rand() / math.sqrt(self.F) for x in range(0, self.F)]
            self.bu[user] = 0
            cnt += len(items)
            for item, rating in items.items():
                if item not in self.Q:
                    self.Q[item] = [np.random.rand() / math.sqrt(self.F) for x in range(0, self.F)]
                    self.bi[item] = 0
        self.mu /= cnt

    # 预测user对item的评分，这里没有使用向量形式
    def predict(self, user, item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(0, self.F)) +\
                self.bu[user] + self.bi[item] + self.mu

    # 有了矩阵之后，进行训练，使用随机梯度下降，训练参数P和Q
    def train(self):
        for step in range(self.max_iter):
            for user, items in self.rating_data.items():
                for item, r_ui in items.items():
                    rhat_ui = self.predict(user, item)  # 得到预测评分
                    e_ui = r_ui - rhat_ui
                    self.bu[user] += self.alpha * (e_ui - self.lmbda * self.bu[user])
                    self.bi[item] += self.alpha * (e_ui - self.lmbda * self.bi[item])

                    # 随机梯度下降更新梯度
                    for k in range(0, self.F):
                        self.P[user][k] += self.alpha * (e_ui * self.Q[item][k] + self.P[user][k])
                        '''这里感觉有不对的地方'''
                        self.Q[item][k] += self.alpha * (e_ui * self.P[user][k] + self.Q[item][k])
            self.alpha *= 0.1  # 每次迭代步数逐步缩小



# 定义数据集，用字典存放数据，避免实际中普遍稀疏的情况
def loadData():
    rating_data={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
           2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
           3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
           4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
           5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
          }
    return rating_data

if __name__ == '__main__':
    # 训练和测试
    rating_data = loadData()
    print(rating_data)
    basicsvd = SVD(rating_data, F=10)
    basicsvd.train()
    for item in ['E']:
        print(item, basicsvd.predict(1, item))