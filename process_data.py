import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(input_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            id_num, text, dem_flag, speaker = [l.strip() for l in line.split("\t")]
            words = set(text.split())
            train_flag = 0
            if speaker in ('REAGAN','JFK'):
                train_flag = 1 
                for word in words:
                    vocab[word] += 1
            if len(text.split()) > 200:
                print speaker
                print sent[:199]
                print "That's a LONG sentence"
                sys.exit(1)
            else:
                if clean_string:
                    text = text.lower()       
                    text = clean_str(text)
                datum  = {"y":int(dem_flag),
                          "id":id_num,
                          "speaker": speaker, 
                          "text": text,                             
                          "num_words": len(text.split()),
                          "split": np.random.randint(0,cv),
                          "train_flag": train_flag}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=100):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float64')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)   
    # in my opinion we should either lemmatize or keep 
    # contractions as they are   
    # string = re.sub(r"\'s", " \'s", string) 
    # string = re.sub(r"\'ve", " \'ve", string) 
    # string = re.sub(r"n\'t", " n\'t", string) 
    # string = re.sub(r"\'re", " \'re", string) 
    # string = re.sub(r"\'d", " \'d", string) 
    # string = re.sub(r"\'ll", " \'ll", string) 
    ## Do we want to remove punctuation?
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]       
    print "loading data...",   
    ## Vocab maps words to counts of how many sentences the word appears in 
    ## revs is a list of dictionary with the following structure
    ##     y : which class it is in (0,1)
    ##     text : the string
    ##     num_words: duh
    ##     cv: which fold it is in (1..10)
    revs, vocab = build_data_cv("./rawdata/output.csv", cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    max_l_ind = np.argmax(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    # print "loading word2vec vectors...",
    #w2v = load_bin_vec(w2v_file, vocab)
    # print "word2vec loaded!"
    # print "num words already in word2vec: " + str(len(w2v))
    ## If a word from the dictionary is not in the word2vec
    ## Add it and initialize with a random 300x1 vector 
    #add_unknown_words(w2v, vocab)
    ## W is an array of vocab_size +1, 300 
    ## The first row contains the zero matrix
    ## word_idx_map maps from words to their index in W 
    #W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    ## Initialize rand_vecs with a random 300x1 vec for each word
    add_unknown_words(rand_vecs, vocab)
    #W2, _ = get_W(rand_vecs)
    W2, word_idx_map = get_W(rand_vecs)
    #cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    cPickle.dump([revs, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    
