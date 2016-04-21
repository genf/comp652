from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
from sklearn.cluster import KMeans #, MiniBatchKMeans
import logging, sys, csv
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def csv_to_array(csv_file):
    master = []  
    names_array = []  
    with open(csv_file) as csvfile:
        name, doc = "",""
        for row in csvfile: 

            if not row.strip():
                continue
            row = row.split("\t")

            if row[2] not in names_array:
                names_array.append(row[2])
            if name == "":
                name = row[2]
            elif name == row[2]:
                doc =" ".join([doc,row[0]])
            elif name != row[2]:
                master.append(doc)
                doc = ""
                name = row[2]
                doc = " ".join([doc,row[0]])
                 
    return master,names_array

if __name__=='__main__':
    
    csv_file = "../rawdata/output.csv"

    speaches, speakers = csv_to_array(csv_file)

    vec_c = CountVectorizer(min_df=2, stop_words='english',strip_accents='ascii') 
    X_freq = vec_c.fit_transform(speaches)

    vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2,ngram_range=(1,1))
    X = vec.fit_transform(speaches)

    pca = PCA(n_components=3)
    pca_freq = pca.fit_transform((X_freq.toarray()))

    pca2 = PCA(n_components=3)
    pca_tfidf = pca.fit_transform((X.toarray()))

    estimators = {'k_means:2clusters': KMeans(n_clusters=2),
              'k_means:4 clusters': KMeans(n_clusters=4),
              'k_means:6 clusters': KMeans(n_clusters=6)}

    for t in [pca_freq,pca_tfidf]:
        fignum = 1
        for name, est in estimators.items():
            fig = plt.figure(fignum, figsize=(4, 3))
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

            plt.cla()

            est.fit(t)
            labels = est.labels_
            print labels, "LABELS"        

            for i in range(len(t)):
                tmp = ":".join([speakers[i].strip(),(str(labels[i]))])
                ax.scatter(t[i, 0], t[i, 1], t[i, 2], c=4)#(labels.astype(np.float)[i]))#, c=labels.astype(np.float))
                ax.text(t[i, 0], t[i, 1], t[i, 2], '%s'%(tmp), size=7, zorder=1, color='k')
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_title("%s"%(name))

            fignum = fignum + 1
    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
 
    plt.show()

    
    
