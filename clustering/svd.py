from encoding import csv_to_array
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.stem import WordNetLemmatizer


if __name__=='__main__':

    exclude_anomalies = True
    lemmatize = False
    csv_file = "../rawdata/output.csv"


    print "Getting speaches...\n"
    speaches, speakers = csv_to_array(csv_file)

    ## exclude anomalous politicians 
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

    print "Creating bow vectors...\n"
    vec_c = CountVectorizer(min_df=2, stop_words='english',strip_accents='ascii') 
    X_freq = vec_c.fit_transform(speaches)

    print "Creating tfidf vectors...\n"
    vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2,ngram_range=(1,1))
    X = vec.fit_transform(speaches)

    ## bow vectors
    print "Applying LSA on bow vectors...\n"
    svd = TruncatedSVD(n_components=3)
    svd_freq = svd.fit_transform(X)

    ## tfidf
    print "Applying LSA on tfidf vectors...\n"
    svd2 = TruncatedSVD(n_components=3)
    svd_tfidf= svd2.fit_transform(X)

    ## kmeans
    print "Running kmeans ...\n"
    estimators = {'k_means: 2 clusters': KMeans(n_clusters=2),
              'k_means:4 clusters': KMeans(n_clusters=4),
              'k_means:6 clusters': KMeans(n_clusters=6)}    
    ## graphing


    #for t in [svd_freq,svd_tfidf]:
    for t in [svd_freq]:
        ##hacky way to add colors (means can't do more than len(colors) clusters but...
        colors = ['b','g','r','c','m','y','k','w']
        print "Computing graphs for %s"%(t)
        fignum = 1
        for name, est in estimators.items():
            fig = plt.figure(fignum, figsize=(4, 3))
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=115)
           # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

            plt.cla()

            est.fit(t)
            labels = est.labels_
            print labels, "LABELS"        


            for i in range(len(t)):
                tmp = ":".join([speakers[i].strip(),(str(labels[i]))])
                ax.scatter(t[i, 0], t[i, 1], t[i, 2], c=colors[labels[i]])
                ax.text(t[i, 0], t[i, 1], t[i, 2], '%s'%(tmp), size=7, zorder=1, color='k')


            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_title("%s"%(name))

            fignum = fignum + 1
        # Plot the ground truth
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=134)
        #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()

        plt.show()