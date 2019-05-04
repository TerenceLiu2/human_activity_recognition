"""

图三

"""


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def ml_model(X_train,y_train,X_test,y_test,model):
    '''
    六种机器学习模型
    '''

    if model == 'SVM with Linear Kernel':
        clf = SVC(kernel='linear')
        clf.fit(X_train,y_train)
    elif model == 'SVM with Radial Basis Kernel':
        clf = SVC(kernel='rbf',C=11)
        clf.fit(X_train, y_train)
    elif model == 'SVM with Polynomial Kernel':
        clf = SVC(kernel='poly',C=11)
        clf.fit(X_train, y_train)
    elif model == 'Gradient Boosted Trees':
        clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
    elif model == 'Linear Discriminant Analysis':
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
    elif model == 'Multinomial Model':
        clf = LogisticRegression(multi_class='multinomial',solver='newton-cg')
        clf.fit(X_train, y_train)

    # 预测结果
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    # 计算error_rate
    train_error = str(round((1-accuracy_score(y_train,y_train_pred))*100,2)) + '%'
    test_error = str(round((1-accuracy_score(y_test,y_test_pred))*100,2) ) + '%'
    return [train_error,test_error]

X_train = []
y_train = []
X_test = []
y_test = []

"""
数据读取与转换
"""
with open('../data/train/X_train.txt') as rf:
    lines = rf.read().split('\n')[:-1]
    for l in lines:
        X_train.append(l.split())

with open('../data/test/X_test.txt') as rf:
    lines = rf.read().split('\n')[:-1]
    for l in lines:
        X_test.append(l.split())

with open('../data/train/y_train.txt') as rf:
    lines = rf.read().split('\n')[:-1]
    for l in lines:
        y_train.append(l)

with open('../data/test/y_test.txt') as rf:
    lines = rf.read().split('\n')[:-1]
    for l in lines:
        y_test.append(l)

X_train = np.array(X_train).astype(np.float64)
y_train = np.array(y_train).astype(np.float64)
X_test = np.array(X_test).astype(np.float64)
y_test = np.array(y_test).astype(np.float64)

col_labels = ['train_errors','test_errors']
row_labels = ['SVM with Linear Kernel','SVM with Radial Basis Kernel','SVM with Polynomial Kernel',
              'Gradient Boosted Trees','Linear Discriminant Analysis','Multinomial Model']
cell_text = []
"""
训练不同的模型并记录error率
"""
for l in row_labels:
    error = ml_model(X_train,y_train,X_test,y_test,l)
    cell_text.append(error)

"""
绘图
"""

fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
table = plt.table(cellText=cell_text,rowLabels=row_labels,colLabels=col_labels,
                  colWidths = [0.1] * 2, loc = 'center', cellLoc = 'center')
table.set_fontsize(30)
table.scale(1.5,2)
plt.show()

