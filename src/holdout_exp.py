'''

图六

'''

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

X_users = {}
y_users = {}

"""
根据用户ID读取数据
"""

with open('../data/train/subject_train.txt') as user_f:
    X = open('../data/train/X_train.txt').read().split('\n')[:-1]
    y = open('../data/train/y_train.txt').read().split('\n')[:-1]
    lines = user_f.read().split('\n')[:-1]
    for i,line in enumerate(lines):
        user_id = line.replace('\n', '')
        try:
            X_users[user_id].append(X[i].split())
            y_users[user_id].append(y[i])
        except:
            X_users[user_id] = []
            y_users[user_id] = []
            X_users[user_id].append(X[i].split())
            y_users[user_id].append(y[i])

with open('../data/test/subject_test.txt') as user_f:
    X = open('../data/test/X_test.txt').read().split('\n')[:-1]
    y = open('../data/test/y_test.txt').read().split('\n')[:-1]
    lines = user_f.read().split('\n')[:-1]
    for i,line in enumerate(lines):
        user_id = line.replace('\n', '')
        try:
            X_users[user_id].append(X[i].split())
            y_users[user_id].append(y[i])
        except:
            X_users[user_id] = []
            y_users[user_id] = []
            X_users[user_id].append(X[i].split())
            y_users[user_id].append(y[i])


error_dict = {}

'''
遍历30个用户取29个为训练,1个为验证计算错误率
'''
for i in range(1,31):
    user_list = list(range(1,31))
    user_list.remove(i)
    X_train = []
    y_train = []

    # 构建训练集
    for user in user_list:
       X_train.extend(X_users[str(user)])
       y_train.extend(y_users[str(user)])

    # 构建测试集
    X_test = X_users[str(i)]
    y_test = y_users[str(i)]

    # 数据预处理
    X_train = np.array(X_train).astype(np.float64)
    y_train = np.array(y_train).astype(np.float64)
    X_test = np.array(X_test).astype(np.float64)
    y_test = np.array(y_test).astype(np.float64)

    # 训练模型
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 计算错误率
    error_dict[i] = round(1-accuracy_score(y_test,y_pred),2)

# 画图
error_dict= sorted(error_dict.items(), key=lambda d:d[1], reverse = True)
plt.bar([str(i[0]) for i in error_dict],[i[1] for i in error_dict],.6,color="#87CEFA")
plt.xticks(list(range(0,30)),[i[0] for i in error_dict])
plt.xlabel('Experiement Participants')
plt.ylabel('Misclassification Rate')
plt.title('Hold one out Experiement')
plt.show()


