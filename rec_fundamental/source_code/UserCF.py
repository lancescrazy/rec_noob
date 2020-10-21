import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 读取数据
def get_data(root_path):
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::', engine='python', names=rnames)
    # os.path.join() 路径拼接文件路径；从第一个/开头的参数拼接，若没有则从./开头的上一个参数拼接，否则顺序拼接
    '''
    pd.read_csv
    filepath_or_buffer:str/pathlib
    sep:指定分隔符，否则逗号分隔
    engine：使用的分析引擎--C或者python
    names：用于结果的列名列表，如果数据文件没列标题行，执行header=None
    '''

    # 分割训练和验证集
    trn_data, val_data, _, _ = train_test_split(ratings, ratings, test_size=0.2)
    '''
    train_test_split()
    train_data, tran_target = ratings
    test_size: 测试集占总样本的百分比
    '''

    # trn_data = trn_data.groupby('user_id')['movie_id'].apply(list).reset_index()
    # =========== 解析trn_data ==========
    p1 = trn_data.groupby('user_id')
    p2 = p1['movie_id']
    p3 = p2.apply(list)
    trn_data = p3.reset_index()
    print('.groupby:\n{}'.format(p1.apply(list)))
    print(".['movie_id']:\n{}".format(p2))
    print('.apply(list):\n{}'.format(p3))
    print('.reset_index():\n{}'.format(trn_data))  # 整理表格（索引）
    val_data = val_data.groupby('user_id')['movie_id'].apply(list).reset_index()

    print('movie_id:\n{}'.format(trn_data['movie_id']))
    trn_user_items = {}
    val_user_items = {}

    # 将数组构造成字典的形式，{user_id: [item_id1, item_id2, ..., item_idn]}
    for user, movies in zip(*(list(trn_data['user_id']), list(trn_data['movie_id']))):
        # zip(*zipped)  # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
        trn_user_items[user] = set(movies)
    # print('trn_user_items:\n{}'.format(trn_user_items))
    for user, movies in zip(*(list(val_data['user_id']), list(val_data['movie_id']))):
        val_user_items[user] = set(movies)

    return trn_user_items, val_user_items





if __name__ == '__main__':
    root_path = './data/ml-1m/'
    trn_user_items, val_user_items = get_data(root_path)
