from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from matplotlib import pyplot
import pandas as pd
testForWord=[]
df = pd.read_csv("suicideTweetData.csv",header=None)
df.columns=["Tweet","Value"]
total = df.iloc[:100]
stop= stopwords.words('english')
for index, row in total.iterrows():
    if(word_tokenize(row['Tweet']) not in stop):
        testForWord.append(list(word_tokenize(row['Tweet'])))
model = Word2Vec(testForWord, min_count=10)
model.save('model.bin')
model.wv.save_word2vec_format('model.txt', binary=False)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.savefig('filename2.png', dpi=3000)