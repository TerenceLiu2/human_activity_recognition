'''

使用PCA与t-SNE降维并可视化

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def all_plot(X, y, m):
    '''
    图一与图二的代码
    '''
    if m == 'pca': # 使用PCA降维
        X_new = PCA(n_components=2).fit_transform(X)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2D projection with PCA')
    elif m == 'tsne': # 使用t-SNE降维
        X_new = TSNE(n_components=2).fit_transform(X)
        plt.xlabel('tSNE1')
        plt.ylabel('tSNE2')
        plt.title('2D projection with tSNE')

    # 使用one-hot向量编码标签
    y_onehot = pd.get_dummies(pd.DataFrame(y)).values

    # 循环每个类别上不同的颜色并打印在图上
    for i in range(6):
        # 找到不同的六个类别
        c_loc = np.where(y_onehot[:, i])
        plt.scatter(X_new[c_loc, 0], X_new[c_loc, 1], s=2, edgecolors=color_dict[i],
                    linewidths=2, label=label_dict[i])

    plt.legend(loc="best", scatterpoints=1)

    plt.show()

def single_plot(X,y,user,type):
    '''
    图七和图八的代码
    '''

    # 编码标签one-hot向量
    y_onehot = pd.get_dummies(pd.DataFrame(y)).values
    # 编码用户one-hot向量
    user_onehot = pd.get_dummies(pd.DataFrame(user)).values
    # TSNE降维
    X_new = TSNE(n_components=2).fit_transform(X)

    #根据数据来源不同的用户上不一样的颜色
    for i in range(30):

        # 用交集找出特定标签特定用户的数据点
        c_loc = np.where(y_onehot[:, type])
        u_loc = np.where(user_onehot[:, i])
        loc = np.intersect1d(c_loc, u_loc)

        plt.scatter(X_new[loc,0], X_new[loc,1], s=2, edgecolors=user_color[i],
                linewidths=2)

    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')

    if type == 4:
        plt.title('2D projection for Standing with tSNE')
    else:
        plt.title('2D projection for Walking with tSNE')

    plt.show()


color_dict = {0:'green',1:'blue',2:'orange',3:'black',4:'red',5:'purple'}
label_dict = {0:'Walking',1:'Walking Upstairs',2:'Walking Downstairs',3:'Sitting',4:'Standing',5:'Laying'}
user_color = {0:'#F0F8FF',2:'#00FFFF',3:'#F0FFFF',4:'#F5F5DC',5:'#FFE4C4',6:'#000000',7:'#FFEBCD',8:'#0000FF',
              9:'#8A2BE2',10:'#A52A2A',11:'#DEB887',12:'#5F9EA0',13:'#7FFF00',14:'#D2691E',15:'#FF7F50',16:'#6495ED',
              17:'#FFF8DC',18:'#DC143C',19:'#00FFFF',20:'#00008B',21:'#008B8B',22:'#B8860B',23:'#A9A9A9',24:'#006400',
              25:'#BDB76B',26:'#8B008B',27:'#556B2F',28:'#FF8C00',29:'#9932CC',1:'#8B0000'}

X = []
y = []
user = []

"""
数据读取
"""

for path in ['../data/train/X_train.txt','../data/test/X_test.txt']:
    with open(path) as rf:
        lines = rf.read().split('\n')[:-1]
        for l in lines:
            X.append(l.split())

for path in ['../data/train/y_train.txt','../data/test/y_test.txt']:
    with open(path) as rf:
        lines = rf.read().split('\n')[:-1]
        for l in lines:
            y.append(l)

for path in ['../data/train/subject_train.txt','../data/test/subject_test.txt']:
    with open(path) as rf:
        lines = rf.read().split('\n')[:-1]
        for l in lines:
            user.append(l)

"""
主函数
"""

all_plot(X,y,'pca')
all_plot(X,y,'tsne')
single_plot(X,y,user,4)
single_plot(X,y,user,0)

