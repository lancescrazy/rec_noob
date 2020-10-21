import pandas as pd
import numpy as np
import math

# 定义数据集，注意:采用字典存放数据，因为实际情况中数据是非常稀疏的
def loadData():
    items={'A': {1: 5, 2: 3, 3: 4, 4: 3, 5: 1},
           'B': {1: 3, 2: 1, 3: 3, 4: 3, 5: 5},
           'C': {1: 4, 2: 2, 3: 4, 4: 1, 5: 5},
           'D': {1: 4, 2: 3, 3: 3, 4: 5, 5: 2},
           'E': {2: 3, 3: 5, 4: 4, 5: 1}
          }
    users={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
           2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
           3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
           4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
           5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
          }
    return items, users

items, users = loadData()
item_df = pd.DataFrame(items).T
user_df = pd.DataFrame(users).T
print('item_df:\n{}'.format(item_df))
print('user_df:\n{}'.format(user_df))

'''计算物品的相似矩阵'''
similarity_matrix = pd.DataFrame(np.zeros((len(items), len(items))), index=['A', 'B', 'C', 'D', 'E'], columns=['A', 'B', 'C', 'D', 'E'])

# 遍历每条物品-用户评分数据
for itemID in items:
    for otheritemID in items:
        vec_item = []  # 定义列表，保存当前两个物品的向量值
        vec_otheritem = []
        # userRatingPairCount = 0  # 两件物品均评过分的用户数
        if itemID != otheritemID:
            for userID in users:  # 遍历用户-物品评分数据
                userRatings = users[userID]

                if itemID in userRatings and otheritemID in userRatings:
                    # userRatingPairCount += 1
                    vec_item.append(userRatings[itemID])
                    vec_otheritem.append(userRatings[otheritemID])
            # 获得相似性矩阵（共现矩阵）
            similarity_matrix[itemID][otheritemID] = np.corrcoef(np.array(vec_item), np.array(vec_otheritem))[0][1]
            # similarity_matrix[itemID][otheritemID] = cosine_similarity(np.array(vec_item), np.array(vec_otheritem))[0][1]
print('similarity_matrix:\n{}'.format(similarity_matrix))


'''得到与物品5相似的前n个物品，计算出最终得分来'''
n = 2
similarity_items = similarity_matrix['E'].sort_values(ascending=False)[:n].index.tolist()
print('similarity_items:\n{}'.format(similarity_items))

base_score = np.mean(list(items['E'].values()))
weighted_scores = 0.
corr_values_sum = 0.
for item in similarity_items:  # ['A', 'D']
    corr_value = similarity_matrix['E'][item]  #  物品之间的相似性
    mean_item_score = np.mean(list(items[item].values()))  # 每个物品的打分平均值
    weighted_scores += corr_value * (users[1][item] - mean_item_score)
    corr_values_sum += corr_value
final_scores = base_score + weighted_scores / corr_values_sum
print('用户Alice对物品5的打分:\n{}'.format(final_scores))
user_df.loc[1]['E'] = final_scores
print('user_df:\n{}'.format(user_df))
