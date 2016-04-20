from gensim.models.word2vec import Word2Vec, BrownCorpus, Text8Corpus
from gensim import matutils
import numpy as np
import os, itertools
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

with open("./rawdata/output.csv") as f:
	full = [l.split("\t") for l in f]
	data = [l[0] for l in full]

if not os.path.exists("./w2v_vecs"):
	brown = BrownCorpus("/home/david/nltk_data/corpora/brown/")
	## Gotta provide total_examples
	text8 =  Text8Corpus("./rawdata/text8")
	sent = itertools.chain(text8, brown)
	model = Word2Vec(sentences=sent, size=100, window=5, min_count=5, workers=4)
	model.init_sims(replace=True)
	model.save("w2v_vecs")
else:
	model = Word2Vec.load("w2v_vecs")

print full[0][0]

s = set([word for word in full[0][0].split(" ")]) - stop
s = s & set(model.vocab.keys())
print s 
print model.most_similar(s)

