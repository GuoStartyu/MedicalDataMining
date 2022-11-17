from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pickle  # 保存模型

warnings.filterwarnings("ignore")


def standardization(np_array):
    """ 离差标准化，(Xi-min(X))/(max(X)-min(X)) """
    min_max_scaler = preprocessing.MinMaxScaler()
    ret = min_max_scaler.fit_transform(np_array)
    return ret


def make_matrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


def rand(min, max):
    """生成区间[min,max]内的随机数"""
    return (max - min) * random.random() + min


def sigmoid(x):
    """激活函数"""
    return 1.0 / (1.0 + math.exp(-x))


def derived_sigmoid(x):
    """激活函数的导数"""
    return x * (1 - x)


class BP:
    """
    反向传播类
    """

    def __init__(self, input_n, hidden_n, output_n):
        self.input_n = input_n  # 初始化结点数
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.input_values = [1.0] * self.input_n  # 输入层神经元输出
        self.hidden_values = [1.0] * self.hidden_n  # 中间层神经元输出
        self.output_values = [1.0] * self.output_n  # 隐藏层神经元输出
        self.input_weights = make_matrix(self.input_n, self.hidden_n)  # 权重
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        self.input_correction = []  # 权重修正
        self.output_correction = []
        self.input_bias = []  # 偏置
        self.output_bias = []

        # 权重矩阵赋初值
        for i in range(self.input_n):
            for j in range(self.hidden_n):
                self.input_weights[i][j] = rand(-0.2, 0.2)
        for i in range(self.hidden_n):
            for j in range(self.output_n):
                self.output_weights[i][j] = rand(-0.2, 0.2)

        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)
        self.input_bias = [0.0] * self.hidden_n
        self.output_bias = [0.0] * self.output_n

    def predict(self, inputs):
        """前向传播"""
        # 输入层
        for i in range(self.input_n - 1):
            self.input_values[i] = inputs[i]
        # 隐藏层
        for i in range(self.hidden_n):
            sum = 0.0
            for j in range(self.input_n):
                sum += self.input_values[j] * self.input_weights[j][i]
            self.hidden_values[i] = sigmoid(sum + self.input_bias[i])
        # 输出层
        for i in range(self.output_n):
            sum = 0.0
            for j in range(self.hidden_n):
                sum += self.hidden_values[j] * self.output_weights[j][i]
            self.output_values[i] = sigmoid(sum + self.output_bias[i])
        return self.output_values[:]

    def back_propagate(self, label, learn, correct):
        """反向传播"""
        # 计算输出层的误差
        output_deltas = [0.0] * self.output_n
        for i in range(self.output_n):
            error = label[i] - self.output_values[i]
            output_deltas[i] = derived_sigmoid(self.output_values[i]) * error
        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_n
        for i in range(self.hidden_n):
            error = 0.0
            for j in range(self.output_n):
                error += output_deltas[j] * self.output_weights[i][j]
            hidden_deltas[i] = derived_sigmoid(self.hidden_values[i]) * error
        # 更新输出层权重
        for i in range(self.hidden_n):
            for j in range(self.output_n):
                change = output_deltas[j] * self.hidden_values[i]
                self.output_weights[i][j] += learn * change + correct * self.output_correction[i][j]
                self.output_correction[i][j] = change
                self.output_bias[j] += learn * change
        # 更新隐藏层权重
        for i in range(self.input_n):
            for j in range(self.hidden_n):
                change = hidden_deltas[j] * self.input_values[i]
                self.input_weights[i][j] += learn * change + correct * self.input_correction[i][j]
                self.input_correction[i][j] = change
                self.input_bias[j] += learn * change

        # 计算样本的均方误差
        error = 0.0
        for i in range(len(label)):
            error += 0.5 * (label[i] - self.output_values[i]) ** 2
        return error

    def train(self, datas, labels, epochs, learn, correct, stop_error=0.01):
        # 进度条
        """训练"""
        switch = {
            0.0: [1, 0],
            1.0: [0, 1]
        }
        for i in tqdm(range(epochs)):
            time.sleep(0.05)
            error = 0.0
            for j in range(len(datas)):
                label = switch[labels[j][0]]  # 标签转换为向量
                data = datas[j]
                self.predict(data)
                error += self.back_propagate(label, learn, correct)
            if error <= stop_error:
                return i + 1
        return epochs

    def test(self, data, x_test, y_test):
        """测试"""
        true_n = 0  # 统计正确预测数量
        pre_proba = []  # 预测概率
        pre_label = []  # 预测结果
        # 测试数据
        for i in range(x_test.shape[0]):
            inputs = x_test[i]
            predict_value = self.predict(inputs)
            pre_proba.append(predict_value)
            label = predict_value.index(max(predict_value))
            if label == y_test[i]:
                true_n += 1
            pre_label.append(label)

        pre_proba = np.array(pre_proba)
        pre_label = np.array(pre_label)
        y_test = y_test.flatten()  # 将二维数组转换为一维数组
        accuracy = true_n / x_test.shape[0]
        # 准确率
        print(f"Accuracy: {accuracy}")
        # 精确率
        print("Precision:", precision_score(y_test, pre_label, average='macro'))
        # 召回率
        print("Recall:", recall_score(y_test, pre_label, average='macro'))
        # F1值
        print("F1-Score:", f1_score(y_test, pre_label, average='macro'))
        # 画ROC曲线
        fpr, tpr, thresholds = roc_curve(y_test, pre_proba[:, 1], pos_label=1)  # pos_label=1表示正样本的标签为1
        plt.plot(fpr, tpr, linewidth=2, label="ROC(AUC=%0.3f)" % roc_auc_score(y_test, pre_proba[:, 1]), color='green')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.legend(loc=4)
        plt.show()


if __name__ == '__main__':
    # 加载数据
    df = pd.read_excel(r"C:\Users\guo\Desktop\课程\医学数据挖掘\实验2-神经网络分析中医药数据实验\症状_瘀血阻络证_data.xlsx")
    data = df.iloc[:, :-1]
    data['target'] = df.iloc[:, -1]  # 添加标签
    data = data.to_numpy()

    # 参数设置
    train_n = math.floor(data.shape[0] * 0.7)  # 训练样本数,7:3
    input_nodes = data.shape[1] - 1  # 输入层节点数
    hidden_nodes = 15  # 隐藏层节点数
    output_nodes = 2  # 输出层节点数
    epochs = 1000  # 训练次数
    learn_rate = 0.001  # 学习率
    correct_rate = 0.1  # 矫正率

    print("---------------------7:3划分训练集和验证集-----------------------")
    # stand_data = standardization(data[:, :data.shape[1]-1])  # 数据标准化
    # data = np.concatenate((stand_data, data[:, data.shape[1]-1:data.shape[1]]), axis=1)  # 拼接标准化后的数据和标签
    x_train, x_test, y_train, y_test = train_test_split(data[:, :data.shape[1] - 1],
                                                        data[:, data.shape[1] - 1:data.shape[1]],
                                                        test_size=data.shape[0] - train_n)

    bp = BP(input_nodes, hidden_nodes, output_nodes)  # 创建BP神经网络
    bp.train(x_train, y_train, epochs, learn_rate, correct_rate)  # 训练数据

    # 加载模型
    # with open('bp_model.pkl', 'rb') as f:
    #     bp = pickle.load(f)
    bp.test(data, x_test, y_test)  # 测试数据

    print("---------------------不划分训练集和验证集-----------------------")
    # 不划分训练集和验证集
    x_train = data[:, :data.shape[1] - 1]
    y_train = data[:, data.shape[1] - 1:data.shape[1]]

    bp = BP(input_nodes, hidden_nodes, output_nodes)  # 创建BP神经网络
    bp.train(x_train, y_train, epochs, learn_rate, correct_rate)  # 训练数据

    bp.test(data, x_train, y_train)  # 测试数据

    print('---------------------------交叉验证----------------------------------')
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(data):
        x_train, x_test = data[train_index, :data.shape[1] - 1], data[test_index, :data.shape[1] - 1]
        y_train, y_test = data[train_index, data.shape[1] - 1:data.shape[1]], data[test_index,
                                                                              data.shape[1] - 1:data.shape[1]]

        bp = BP(input_nodes, hidden_nodes, output_nodes)
        bp.train(x_train, y_train, epochs, learn_rate, correct_rate)
        bp.test(data, x_test, y_test)
