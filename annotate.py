from __future__ import division
import pandas as pd
from gensim.models.word2vec import Word2Vec
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math

tqdm.pandas(desc="progress-bar")
stop= stopwords.words('english')

def tokenize(tweet):
	word_tokens = tweet.split(' ')	
	tokens = [w for w in word_tokens if not w in stop] 
	return tokens

data = pd.read_csv('../datasets/suicideTweetData2.csv',header = None)
data.columns = ["Tweet_text","Class_Label"]

data["tokens"] = data["Tweet_text"].progress_map(tokenize)
x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),np.array(data.Tweet_text), test_size=0.0)

tweet_w2v = Word2Vec(size=250, min_count=10)
tweet_w2v.build_vocab([x for x in tqdm(x_train)])
tweet_w2v.train([x for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)

def vector(tokens):
	vec = np.zeros(250)
	count = 0
	for token in tokens:
		try:
			vec = vec + tweet_w2v[token]
			count += 1
		except KeyError:
			continue
	if count != 0:
		return np.divide(vec,count)
	else:
		return vec
data["vector"] = data["tokens"].progress_map(vector)

class heapobj:
	def __init__(self,dis,y):
		self.eq_dist=dis
		self.y_val=y

def predict(X_train,X_test,Y_train,Y_test,k1,pos):
	for i in range(X_test.shape[0]):
		L=[]
		KN=[]
		test_row=list(X_test.iloc[i])
		actual=Y_test.iloc[i]
		dist=0
		freq0=0
		freq1 = 0
		for j in range(X_train.shape[0]):
			curr_row=list(X_train.iloc[j])
			val=Y_train.iloc[j]
			for k in range(250):
				dist+=pow((curr_row[k]-test_row[k]),2)
			dist=math.sqrt(dist)
			L.append(heapobj(dist,val))
		L=sorted(L,key=lambda x:x.eq_dist)
		for m in range(k1):
			if L[m].y_val == 0:
				freq0 = freq0 + 1
			else:
				freq1 = freq1 + 1
		if freq0 >= freq1:
			df.iloc[pos+i,1]= 0
		else:
			df.iloc[pos+i,1]= 1
df = data

X_train=df.iloc[0:500,3]
Y_train=df.iloc[0:500,1]
X_test = df.iloc[501:1000,3]
Y_test = df.iloc[501:1000,1]
predict(X_train,X_test,Y_train,Y_test,5,501)
print(5)
X_train=df.iloc[0:1000,3]
Y_train=df.iloc[0:1000,1]
X_test = df.iloc[1001:2000,3]
Y_test = df.iloc[1001:2000,1]
predict(X_train,X_test,Y_train,Y_test,7,1001)
print(7)

X_train=df.iloc[0:2000,3]
Y_train=df.iloc[0:2000,1]
X_test = df.iloc[2001:5000,3]
Y_test = df.iloc[2001:5000,1]
predict(X_train,X_test,Y_train,Y_test,13,2001)
print(13)

X_train=df.iloc[0:5000,3]
Y_train=df.iloc[0:5000,1]
X_test = df.iloc[5001:10000,3]
Y_test = df.iloc[5001:10000,1]
predict(X_train,X_test,Y_train,Y_test,17,5001)
print(17)

X_train=df.iloc[0:10000,3]
Y_train=df.iloc[0:10000,1]
X_test = df.iloc[10001:df.shape[0],3]
Y_test = df.iloc[10001:df.shape[0],1]
predict(X_train,X_test,Y_train,Y_test,23,10001)
print(23)

df.to_csv("final_data.csv",sep=',', encoding='utf-8')
print("ANNOTATION COMPLETE")


