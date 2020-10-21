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



'''计算用户相似性矩阵'''
similarity_matrix = pd.DataFrame(np.zeros((len(users), len(users))), index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
# np.zeros(shape, dtype=float, order='C')
# pd.DataFrame()函数解析(github.com/lancescrazy/rec_noob/dataframe函数解析
# 遍历每条用户-物品评分数据
for userID in users:
    for otheruserID in users:
        vec_user = []
        vec_otheruser = []
        if userID != otheruserID:
            # print('userID:{}'.format(userID))
            # print('otheruserID:{}'.format(otheruserID))
            for itemID in items:
                itemRatings = items[itemID]
                if userID in itemRatings and otheruserID in itemRatings:
                    vec_user.append(itemRatings[userID])
                    vec_otheruser.append(itemRatings[otheruserID])
                # 这里可以获得相似性矩阵（共现矩阵）
            corrcoef = np.corrcoef(np.array(vec_user), np.array(vec_otheruser))  # corrcoef 的数据结构是什么样子的，打印观察
            # print('corrcoef:{}; corrcoef_type:{}'.format(corrcoef, type(corrcoef)))
            similarity_matrix[userID][otheruserID] = corrcoef[0][1]

print('similarity_matrix:\n{}\n'.format(similarity_matrix))


'''计算前n个相似的用户'''
n = 2
similarity_users = similarity_matrix[1].sort_values(ascending=False)[:n].index.tolist()  # [2, 3] 也就是用户1和用户2
# # ============================== 解析 =================================
# p1 = similarity_matrix[1]
# p2 = p1.sort_values(ascending=False)
# p3 = p2.index
# # 取出表的第一行
# print('1st process: 1column of similarity_matrix:\n{}; type_p1: {}'.format(p1, type(p1)))
# # 按照值从大到小排序
# # DataFrame.sort_values(by='##',axis=0,ascending=True,inplace=False,na_position='last')
# # by 指定列名(axis = 0或'index') 或者索引(axis = 1或'columns')
# # axis	若axis=0或’index’，则按照指定列中数据大小排序；若axis=1或’columns’，则按照指定索引中数据大小排序，默认axis=0
# # ascending	是否按指定列的数组升序排列，默认为True，即升序排列
# # inplace	是否用排序后的数据集替换原来的数据，默认为False，即不替换
# # na_position	{‘first’,‘last’}，设定缺失值的显示位置
# print('2nd process: sort_values:\n{}; type_p2: {}'.format(p2, type(p2)))
# # .index 将Series类转化为int64index,即整数型索引
# # .tolist() 将索引转换为列表
# print('3th process: \n{}; type_p3: {}'.format(p3, type(p3)))
# 输出结果
print('similarity_users:\n{}'.format(similarity_users))


'''计算最终得分'''
base_score = np.mean(list(users[1].values()))
weighted_scores = 0.
corr_values_sum = 0.
for user in similarity_users:  # [2, 3]
    corr_value = similarity_matrix[1][user]  # 两个用户间的相似性
    mean_user_score = np.mean(list(users[user].values()))  # 用户的打分平均值
    weighted_scores += corr_value * (users[user]['E'] - mean_user_score)
    corr_values_sum += corr_value
final_scores = base_score + weighted_scores / corr_values_sum
print('用户Alice对物品5的打分：{}'.format(final_scores))
user_df.loc[1]['E'] = final_scores
print(user_df)