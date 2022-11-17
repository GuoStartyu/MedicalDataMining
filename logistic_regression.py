"""
logistic回归
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

data = pd.read_excel(r"C:\Users\guo\Desktop\课程\医学数据挖掘\实验1-逻辑回归分析中医药数据\症状_瘀血阻络证_data.xlsx")
lr = LogisticRegression()  # 建立逻辑回归模型
X = data.iloc[:, :-1]  # 特征
Y = data.iloc[:, -1]  # 标签

print('---------------------------不划分训练集和测试集---------------------------------')
lr.fit(X, Y)  # 训练模型
# 准确率
print("Accuracy:", lr.score(X, Y))
# 精确率
print("Precision:", precision_score(Y, lr.predict(X)))
# 召回率
print("Recall:", recall_score(Y, lr.predict(X)))
# F1值
print("F1-Score:", f1_score(Y, lr.predict(X)))
# 画出ROC曲线
fpr, tpr, thresholds = roc_curve(Y, lr.predict_proba(X)[:, 1])
plt.plot(fpr, tpr, linewidth=2, label="ROC(AUC=%0.3f)" % roc_auc_score(Y, lr.predict_proba(X)[:, 1]), color="green")
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()
# AUC
# print("AUC:", roc_auc_score(Y, lr.predict(X)))


print('---------------------------划分训练集和测试集----------------------------------')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)  # 划分训练集和测试集
lr.fit(x_train, y_train)
# 准确率
print("Accuracy:", lr.score(x_test, y_test))
# 精确率
print("Precision:", precision_score(y_test, lr.predict(x_test)))
# 召回率
print("Recall:", recall_score(y_test, lr.predict(x_test)))
# F1值
print("F1-Score:", f1_score(y_test, lr.predict(x_test)))
# 画出ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, linewidth=2, label="ROC(AUC=%0.3f)" % roc_auc_score(y_test, lr.predict_proba(x_test)[:, 1]), color="green")
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()

print('---------------------------交叉验证----------------------------------')
# 5-fold cross-validation

scores = cross_val_score(lr, X, Y, cv=5)
print('scores:', scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 精确率
print("Precision:", cross_val_score(lr, X, Y, cv=5, scoring='precision').mean())
# 召回率
print("Recall:", cross_val_score(lr, X, Y, cv=5, scoring='recall').mean())
# F1值
print("F1-Score:", cross_val_score(lr, X, Y, cv=5, scoring='f1').mean())
# 画出ROC曲线

y_scores = cross_val_predict(lr, X, Y, cv=5, method='decision_function')
fpr, tpr, thresholds = roc_curve(Y, y_scores)
plt.plot(fpr, tpr, linewidth=2, label='ROC(AUC=%0.3f)' % cross_val_score(lr, X, Y, cv=5, scoring='roc_auc').mean(),
         color='green')
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()
# AUC
# print("AUC:", cross_val_score(lr, X, Y, cv=5, scoring='roc_auc').mean())
