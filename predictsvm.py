import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score
import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
df = pd.read_csv("final_data.csv")

def change(x):
	x = x[1:len(x)-2]
	x = x.replace('\n',' ').split()
	for i in range(len(x)):
		x[i] = float(x[i])
	return x
df["final"] = df["vector"].progress_map(change)
list = []
for i in range(df.shape[0]):
	list.append(df.iloc[i][5])

X = list
y = df.iloc[:,2]

# classifier one :support vector machine
clf = svm.SVC(kernel='rbf', C = 1.0)
print("TRAINING USING SVM....")

#splitting the data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.12)
X_folds = np.array_split(X, 10)
y_folds = np.array_split(y, 10)
scores = []
for k in range(10):
	X_train = X_folds.copy()
	X_test  = X_train.pop(k)
	X_train = np.concatenate(X_train)
	y_train = y_folds.copy()
	y_test  = y_train.pop(k)
	y_train = np.concatenate(y_train)
	scores.append(clf.fit(X_train, y_train).score(X_test, y_test))

clf.fit(X_train1,y_train1)
print("TRAINING USING SVM DONE!")
clf_pred=clf.predict(X_test1)

#printing parameters
precision,recall,fscore,support = score(y_test1,clf_pred)
print("Classes not predicted are:",set(y_test1)-set(clf_pred))
print("USING SVM")
print('precision: {}'.format(precision))
print('recall:{}'.format(recall))
print('fscore: {}'.format(fscore))
temp = 0
for score in scores:
	temp += score
acc1 = temp / len(scores) * 100 
print("acc using SVM is ",acc1)


