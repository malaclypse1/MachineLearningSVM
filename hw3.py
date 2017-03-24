import pandas as pd
import sklearn.preprocessing
import sklearn.utils
import sklearn.model_selection
import sklearn.svm as svm
import numpy as np
import itertools
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest

#read data file, split it into equal training and test sets
#scale the features to the training set
data = pd.read_csv("spambase.data", header=None, index_col=57)
data = sklearn.utils.shuffle(data)
X_train, X_test,y_train,y_test = sklearn.model_selection.train_test_split(data,data.index.values,test_size=0.5)
scalar = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#classifications should be 1 or -1, not 1 or 0
y_train = [-1 if elem == 0 else elem for elem in y_train]
y_test = [-1 if elem == 0 else elem for elem in y_test]

#check balance of training data to test data
#sum classification: 0 = perfectly balanced between pos and neg
#convert to % of overall set balance
#ideally, train balance = test balance
train_balance = sum(y_train)
test_balance = sum(y_test)
train_balance,test_balance = train_balance*100/(train_balance+test_balance),test_balance*100/(train_balance+test_balance)
print("train balance %d%%" % train_balance)
print("test balance %d%%" % test_balance)

#experiment 1
#train SVM
print("Experiment 1: linear kernel SVM over all features")
svc = svm.SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_score = svc.decision_function(X_test)
acc = sm.accuracy_score(y_test, y_pred)
prec = sm.precision_score(y_test, y_pred)
rec = sm.recall_score(y_test, y_pred)
print("Accuracy %f, precision %f, recall %f" % (acc, prec, rec))

fpr, tpr, _ = sm.roc_curve(y_test, y_score)
roc_auc = sm.auc(fpr, tpr)

plt.figure(1)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - experiment 1')
plt.legend(loc="lower right")
plt.savefig("ex1.png")

#experiment 2
#get order of weights by |w|
print("Experiment 2: feature selection with highest weighted features")
abs_w = np.absolute(svc.coef_[0]);
w_sorted = np.argsort(abs_w)
w_sorted = np.flipud(w_sorted)
#get the 5 best features
#tail -n 57 spambase.names > spambase.featurekey
print("top 5 features:")
with open("spambase.featurekey") as fp:
    for i, line in enumerate(fp):
        if i in w_sorted[:5]:
            print(line)

#array for plotting data 
#row 0 = m
#row 1 = best features accuracy
#row 2 = random features accuracy (for exp 3)
acc2=np.zeros((3,56))
for k in range(2,58):
    X2_train = X_train[:, w_sorted[:k]]
    X2_test = X_test[:, w_sorted[:k]]
    svc.fit(X2_train, y_train)
    y2_pred = svc.predict(X2_test)
    acc2[0,k-2] = k
    acc2[1,k-2] = sm.accuracy_score(y_test, y2_pred)
plt.figure(2)
plt.plot(acc2[0],acc2[1], color='darkorange', label='best features')
plt.xlim([0.0, 57.0])
plt.ylim([0.4, 1.0])
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.title('accuracy vs features - experiments 2 & 3')

#experiment 3
print("Experiment 3: feature selection with random features")
for k in range(2,58):
    w_random = np.random.choice(57, size=k, replace=False)
    X3_train = X_train[:, w_random]
    X3_test = X_test[:, w_random]
    svc.fit(X3_train, y_train)
    y3_pred = svc.predict(X3_test)
    acc2[2,k-2] = sm.accuracy_score(y_test, y3_pred)
plt.plot(acc2[0],acc2[2], color='navy', label='random features')
plt.legend(loc="lower right")
plt.savefig("ex2.png")
plt.show()
