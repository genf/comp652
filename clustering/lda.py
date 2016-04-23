import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
from pca import csv_to_array
import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords

if __name__=='__main__':
    
   
    csv_file = "../rawdata/output.csv"
    lemmatize = True
    exclude_anomalies = True

    print "Getting speaches...\n"
    speaches, speakers = csv_to_array(csv_file)

    if exclude_anomalies:
        speaks = []
        speech = []
        for speaker, text in zip(speakers, speaches):
            "Excluding anomalies...\n"
            if speaker.strip() not in ('JFK','REAGAN', 'WEBB', 'CHAFEE', 'JINDAL', 'PERRY'):
                speaks.append(speaker)
                speech.append(text)

        speaches=speech
        speakers = speaks

    print len(speaches)
    print speakers

    
    if lemmatize:
        ## lematize
        lmtzr = WordNetLemmatizer()
        for i in range(len(speaches)):
            print  "Lemmatizing speach %d...\n"%(i)
            arr = speaches[i].split()
            new_str = ""
            for w in arr:
                w = unicode(w,'utf-8')
                new_str = " ".join([new_str,lmtzr.lemmatize(w)])

            speaches[i] = new_str
    

    documents = speaches

    stoplist = set(stopwords.words('english'))
    texts = [[word for word in document.lower().split() if word not in stoplist]
              for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
        for text in texts]

    ## create corpus
    dictionary = corpora.Dictionary(texts)

    ## create bow vectors for corpus
    print "Creating bow vectors for corpus...\n"
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print corpus

    print "Creating tfidf vectors for corpus...\n"
    tfidf = models.TfidfModel(corpus) ## initialize model
    ## create tfidf vectors for corpus
    corpus_tfidf = tfidf[corpus]

    print "Applying LDA...\n"
    ## latent dirichlet allocation on bow
    lda_bow = models.LdaModel(corpus, id2word=dictionary, num_topics=5, iterations=100)
    
    ## latent dirichlet allocation on tfidf
    lda_tfidf = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=5, iterations=500)


#    print "running print_topics...\n"
#    for i in range(0,lda_tfidf.num_topics-1):
#        lda_tfidf.print_topics(i)

    print "\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
    print "productive information: \n"

    _ = lda_tfidf.print_topics(-1)

