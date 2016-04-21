from encoding import csv_to_array
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if __name__=='__main__':
    
    csv_file = "../rawdata/output.csv"

    speaches, speakers = csv_to_array(csv_file)

    vec_c = CountVectorizer(min_df=2, stop_words='english',strip_accents='ascii') 
    X_freq = vec_c.fit_transform(speaches)

    vec = TfidfVectorizer(stop_words='english',strip_accents='ascii',min_df=2,ngram_range=(1,1))
    X = vec.fit_transform(speaches)

    ## bow vectors
    svd = TruncatedSVD(n_components=3)
    svd_freq = svd.fit_transform(X)

    ## tfidf
    svd2 = TruncatedSVD(n_components=3)
    svd_tfidf= svd2.fit_transform(X)

    ## kmeans
    estimators = {'k_means_3': KMeans(n_clusters=3),
              'k_means_8': KMeans(n_clusters=8),
              'k_means_15': KMeans(n_clusters=15),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}
    
    ## graphing

    for t in [svd_freq,svd_tfidf]:
	    fignum = 1
	    for name, est in estimators.items():
	        fig = plt.figure(fignum, figsize=(4, 3))
	        plt.clf()
	        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	        plt.cla()

	        est.fit(t)
	        labels = est.labels_
	        print labels, "LABELS"        

	        for i in range(len(t)):

	            tmp = " ".join([speakers[i].strip(),(str(labels[i]))])

	            ax.scatter(t[i, 0], t[i, 1], t[i, 2], c=labels[i].astype(np.float))#, c=labels.astype(np.float))
	            ax.text(t[i, 0], t[i, 1], t[i, 2], '%s'%(tmp), size=5, zorder=1,  
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