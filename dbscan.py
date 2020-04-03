import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score

training_dataset = np.load('training_dataset.npy').T


eps_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.1, 6.2, 6.8, 7]
min_samples_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


for eps in eps_range:
    for min_sample in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(training_dataset)
        cluster_labels = dbscan.labels_
        n_clusters = len(set(cluster_labels))
        print('number of clusters :{}'.format(n_clusters))
        if n_clusters > 1:
            silhouette_avg = silhouette_score(training_dataset, cluster_labels)
            vcr_avg = calinski_harabasz_score(training_dataset, cluster_labels)
            print("For n_clusters: {}, VCR_avg: {},  the average of sitlhouette: {}, eps: {}, min_samples: {}".format(n_clusters, vcr_avg, silhouette_avg, eps, min_sample))


inertia = {}
silhouette = {}

