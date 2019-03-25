import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score
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
temp = 0

#classifier two logistic regression
logreg = LogisticRegression()
print("TRAINING USING LR....")
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.10)
#splitting the data
X_folds = np.array_split(X, 10)
y_folds = np.array_split(y, 10)
scores2 = []
for k in range(10):
	X_train = X_folds.copy()
	X_test  = X_train.pop(k)
	X_train = np.concatenate(X_train)
	y_train = y_folds.copy()
	y_test  = y_train.pop(k)
	y_train = np.concatenate(y_train)
	scores2.append(logreg.fit(X_train, y_train).score(X_test, y_test))

logreg.fit(X_train1,y_train1)
print("TRAINING USING LR DONE!")
lr_pred = logreg.predict(X_test1)

precision,recall,fscore,support = score(y_test1,lr_pred)
print("Classes not predicted are:",set(y_test1)-set(lr_pred))
print("USING LR")
print('precision: {}'.format(precision))
print('recall:{}'.format(recall))
print('fscore: {}'.format(fscore))
for score in scores2:
	temp += score
	acc2 = temp / len(scores2) * 100 
print("acc using LR is ",acc2)