from sklearn.feature_extraction.text import TfidfVectorizer
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

    vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2,ngram_range=(1,1))
    X = vec.fit_transform(speaches)

    estimators = {'k_means_3': KMeans(n_clusters=3),
              'k_means_8': KMeans(n_clusters=8),
              'k_means_15': KMeans(n_clusters=15)}
#              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
 #                                             init='random')}
    
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

    
    
