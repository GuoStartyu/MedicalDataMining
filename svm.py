import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict

# 加载数据
train = pd.read_csv(r"C:\Users\guo\Desktop\课程\医学数据挖掘\实验3-支持向量机分析乳腺癌数据实验\breast-cancer-train.csv").iloc[:, 1:]
test = pd.read_csv(r"C:\Users\guo\Desktop\课程\医学数据挖掘\实验3-支持向量机分析乳腺癌数据实验\breast-cancer-test.csv").iloc[:, 1:]

x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

# 用线性核函数建立支持向量机模型
model = svm.SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test))
# 精确率
print("Precision:", precision_score(y_test, model.predict(x_test)))
# 召回率
print("Recall:", recall_score(y_test, model.predict(x_test)))
# F1值
print("F1-Score:", f1_score(y_test, model.predict(x_test)))
# 画出ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, linewidth=2, label="ROC(AUC=%0.3f)" % roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
         color="green")
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()

# 感知机算法
model = Perceptron()
model.fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test))
# 精确率
print("Precision:", precision_score(y_test, model.predict(x_test)))
# 召回率
print("Recall:", recall_score(y_test, model.predict(x_test)))
# F1值
print("F1-Score:", f1_score(y_test, model.predict(x_test)))
# 画出ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, model.predict(x_test))
plt.plot(fpr, tpr, linewidth=2, label="ROC(AUC=%0.3f)" % roc_auc_score(y_test, model.predict(x_test)), color="green")
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()

# 拼接数据，用于5折交叉验证
x = x_train.append(x_test)
y = y_train.append(y_test)

# 5折交叉验证
# 5-fold cross-validation
model = svm.SVC(kernel='linear', probability=True)
scores = cross_val_score(model, x, y, cv=5)
print('scores:', scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 精确率
print("Precision:", cross_val_score(model, x, y, cv=5, scoring='precision').mean())
# 召回率
print("Recall:", cross_val_score(model, x, y, cv=5, scoring='recall').mean())
# F1值
print("F1-Score:", cross_val_score(model, x, y, cv=5, scoring='f1').mean())
# 画出ROC曲线

y_scores = cross_val_predict(model, x, y, cv=5, method='decision_function')
fpr, tpr, thresholds = roc_curve(y, y_scores)
plt.plot(fpr, tpr, linewidth=2, label='ROC(AUC=%0.3f)' % cross_val_score(model, x, y, cv=5, scoring='roc_auc').mean(),
         color='green')
plt.xlabel('FPR')  # False Positive Rate,假阳性率
plt.ylabel('TPR')  # True Positive Rate,真阳性率
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()
