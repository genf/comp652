from gensim.models.word2vec import Word2Vec, BrownCorpus, Text8Corpus
import os, pdb, itertools

word_count = 0
sent_count = 0
for f in os.listdir("./rawdata/training-monolingual.tokenized.shuffled"):
	if not os.path.isfile("./rawdata/training-monolingual.tokenized.shuffled/" + f):
		continue
	with open("rawdata/training-monolingual.tokenized.shuffled/"+f,"r") as file:
		for line in file:
			sent_count += 1		
			word_count += len(line.strip().split(" "))
brown = BrownCorpus("/home/david/nltk_data/corpora/brown/")
        ## Gotta provide total_examples
text8 =  Text8Corpus("./rawdata/text8")
sent = itertools.chain(text8, brown)
for snt in sent:
	sent_count += 1 
	word_count += len(snt)


print "SENTENCE COUNT " + str(sent_count)
print "WORD COUNT " + str(word_count)
