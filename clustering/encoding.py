from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans #, MiniBatchKMeans
import logging, sys, csv, re
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import defaultdict

def csv_to_array(csv_file):
    master = []  
    names_array = []  
    with open(csv_file) as csvfile:
        name, doc = "",""
        for row in csvfile: 

            if not row.strip():
                continue
            row = [r.strip() for r in row.split("\t")]

            if row[3] not in names_array:
                names_array.append(row[3])
            if name == "":
                name = row[3]
            elif name == row[3]:
                doc =" ".join([doc,row[1]])
            elif name != row[3]:
                master.append(doc)
                doc = ""
                name = row[3]
                doc = " ".join([doc,row[1]])
    master = [clean_str(s) for s in master]     
    return master,names_array

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"applause", " ", string)   

    string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)   
    # in my opinion we should either lemmatize or keep 
    # contractions as they are   
    string = re.sub(r"\'s", "s", string) 
    string = re.sub(r"\'ve", "ve", string) 
    string = re.sub(r"n\'t", "nt", string) 
    string = re.sub(r"\'re", "re", string) 
    string = re.sub(r"\'d", " d", string) 
    string = re.sub(r"\'ll", "ll", string) 
    ## Do we want to remove punctuation?
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    return string.strip() if TREC else string.strip().lower()

if __name__=='__main__':
    
    csv_file = "../rawdata/output.csv"

    speaches, speakers = csv_to_array(csv_file)
    vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2,ngram_range=(1,1))
    count_vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2, ngram_range=(1,2))
    Xp = count_vec.fit_transform(speaches)
    oldies = defaultdict(float)
    new = defaultdict(float)
    fnames = np.array(count_vec.get_feature_names()).reshape((-1,1))
    num_new = len(speakers) - 2
    num_old = 2
    # for cnt, row in enumerate(Xp):
    #     # import pdb
    #     # pdb.set_trace()
    #     row = np.array(row.toarray())
    #     top = row.argsort().ravel()[-100:][::-1]

    #     flat = [x[0] for x in fnames[top].tolist()]
    #     for word in flat:
    #         if speakers[cnt] in ('REAGAN','JFK'):
    #             oldies[word]+=1
    #         else:
    #             new[word]+=1
    #     # print speakers[cnt]
    #     # print fnames[top]
    #     # print "\n\n"
    # print "oldies"
    # for word, cnt in oldies.items():
    #     if word in new:
    #         continue
    #     print word + " " +str(cnt)
    # print "\n\n\n"
    # print "new guys"
    # for word, cnt in new.items():
    #     if cnt> float(num_new)/5 and not word in oldies:
    #         print word + " " +str(cnt)
    # print "\n\n\n"
    # inter = set(oldies.keys()) & set(new.keys())
    # print inter

    # sys.exit(0)
    X = vec.fit_transform(speaches)


    estimators = {'k_means_3': KMeans(n_clusters=3),
              'k_means_8': KMeans(n_clusters=8),
              'k_means_15': KMeans(n_clusters=15)}
#              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
 #                                             init='random')}
    speaks = []
    speeches = []
    for speaker, text in zip(speakers, speaches):
        if speaker.strip() not in ('JFK','REAGAN', 'WEBB', 'CHAFEE', 'JINDAL', 'PERRY'):
            speaks.append(speaker)
            speeches.append(text)
    X = vec.fit_transform(speeches)
    clust = 5
    est = KMeans(n_clusters=clust, n_init=10)
    est.fit(X)
    labels = [[] for _ in range(clust)]
    for speaker, label in zip(speaks,est.labels_):
        labels[label].append(speaker)
    for label in labels:
        print label
        print "\n\n"
    sys.exit(0)
    fignum = 1
    for name, est in estimators.items():
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        pca = PCA(n_components=3)
        X_reduce = pca.fit_transform((X.toarray()))

        est.fit(X_reduce)
        labels = est.labels_
        print labels, "LABELS"        

        for i in range(len(X_reduce)):

            tmp = " ".join([speakers[i]," ", " ",(str(labels[i]))])

            ax.scatter(X_reduce[i, 0], X_reduce[i, 1], X_reduce[i, 2], c=labels[i].astype(np.float))#, c=labels.astype(np.float))
            ax.text(X_reduce[i, 0], X_reduce[i, 1], X_reduce[i, 2], '%s'%(tmp), size=5, zorder=1,  
 color='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        fignum = fignum + 1
    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
 
    plt.show()

    
    
