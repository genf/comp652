from gensim.models.word2vec import Word2Vec, BrownCorpus, Text8Corpus
from gensim.utils import to_unicode
import os, itertools

class NewsIter(object):
	
	def __init__(self, path):
		if not path.endswith("/"):
			path += "/"
		self.path = path

	def __iter__(self):
		for fname in os.listdir(self.path):
			fname = os.path.join(self.path,fname)
			if not os.path.isfile(fname):
				continue
			with open(fname,"r") as f:
				for line in f:
					words = to_unicode(line).strip().lower().split(" ")
					## TODO remove punct
					words = [word for word in words if word.strip()]
					if not words:
						continue
					yield words

with open("wordcounts.txt","r") as cnt_file:
	lines = cnt_file.read().split("\n")
	sent_count = int(lines[0].strip().split(" ")[-1])
	word_count = int(lines[1].strip().split(" ")[-1])
	print "Sentences %s , words %s" %(sent_count, word_count)
	

if not os.path.exists("./w2v_vecs"):
	brown = BrownCorpus("/home/david/nltk_data/corpora/brown/")
	## Gotta provide total_examples
	text8 =  Text8Corpus("./rawdata/text8")
	news = NewsIter("./rawdata/training-monolingual.tokenized.shuffled")
	sent = itertools.chain(text8, brown, news)
	model = Word2Vec(sentences=sent, size=100, window=5, min_count=5, workers=4, max_vocab_size=10000000)
	model.init_sims(replace=True)
	model.save("w2v_vecs")
else:
	model = Word2Vec.load("w2v_vecs")


print model.doesnt_match("breakfast cereal dinner lunch".split())
print model.similarity('woman', 'man')
print model.most_similar(positive=['woman', 'king'], negative=['man'])
