import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import calinski_harabasz_score
from kmodes.kmodes import KModes
import kmedoids
from sklearn.metrics.pairwise import pairwise_distances

training_dataset = np.load('training_dataset.npy').T


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#range_n_clusters = [2, 5]

inertia = {}
silhouette = {}

D = pairwise_distances(training_dataset, metric='euclidean')

for n_clusters in range_n_clusters:
    print('number of clusters :{}'.format(n_clusters))

    M, C = kmedoids.kMedoids(D, n_clusters)
    labels = np.zeros(1173)

    for label in C:
        #print(label)
        for point_idx in C[label]:
            labels[point_idx] = label


    silhouette_avg = silhouette_score(training_dataset, labels)
    vcr_avg = calinski_harabasz_score(training_dataset,labels)
    silhouette[n_clusters] = silhouette_avg

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", vcr_avg)

    #sample_silhouette_values = silhouette_samples(training_dataset, cluster_labels)

plt.figure()
plt.plot(list(inertia.keys()), list(inertia.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
#plt.show()

plt.figure()
plt.plot(list(silhouette.keys()), list(silhouette.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette")
#plt.show()