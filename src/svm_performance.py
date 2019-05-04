"""

图四与图五

"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt

label_dict = {0: 'Walking', 1: 'Walking Upstairs', 2: 'Walking Downstairs', 3: 'Sitting', 4: 'Standing',
                  5: 'Laying'}
label_list = list(label_dict.values())

def train_with_sizes(X_train, y_train, X_test, y_test):
    """
    图4代码,不同训练数据的大小从500-7000
    """
    #设置不同大小
    training_size = [i for i in range(500, 7500, 500)]
    train_e_list = []
    test_e_list = []

    for s in training_size:
        # 截取不同大小的训练集
        X_train_tmp = np.array(X_train[:s]).astype(np.float64)
        y_train_tmp = np.array(y_train[:s]).astype(np.float64)
        X_test = np.array(X_test).astype(np.float64)
        y_test = np.array(y_test).astype(np.float64)
        # 训练模型
        clf = SVC(kernel='linear')
        clf.fit(X_train_tmp, y_train_tmp)
        # 验证模型计算错误率
        y_train_pred = clf.predict(X_train_tmp)
        y_test_pred = clf.predict(X_test)
        train_error = 1 - accuracy_score(y_train_tmp, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        train_e_list.append(train_error)
        test_e_list.append(test_error)

    print(train_error)
    # 画图
    plt.plot(training_size, train_e_list, color='green', label='train errors')
    plt.plot(training_size, test_e_list, color='red', label='test errors')

    # 转换为百分比
    def to_percent(temp, position):
        return '%1.1f' % (100 * temp) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.legend()
    plt.xlabel('Traning Size')
    plt.ylabel('Error Rate')

    plt.show()

def report(X_train,y_train,X_test,y_test):
    """
    图五的代码
    """
    # 预处理数据
    X_train = np.array(X_train).astype(np.float64)
    y_train = np.array(y_train).astype(np.float64)
    X_test = np.array(X_test).astype(np.float64)
    y_test = np.array(y_test).astype(np.float64)
    # 训练模型
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    # 预测并计算混淆矩阵
    y_test_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test,y_test_pred).tolist()
    report_dict = classification_report(y_test,y_test_pred,output_dict=True,labels=[1,2,3,4,5,6])
    tmp_list = []
    for i in range(1,7):
        tmp_list.append(round(report_dict[str(i)]['precision'],2))
    c_matrix.append(tmp_list)

    # 画图
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    table = plt.table(cellText=c_matrix, rowLabels=label_list+['Accuracy'], colLabels=label_list,
                      colWidths=[0.15] * 6, loc='center', cellLoc='center')
    table.set_fontsize(100)
    table.scale(1.1, 1.5)
    plt.show()


X_train = []
y_train = []
X_test = []
y_test = []

'''
数据读取
'''

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

'''
主函数
'''
train_with_sizes(X_train,y_train,X_test,y_test)
# report(X_train,y_train,X_test,y_test)
