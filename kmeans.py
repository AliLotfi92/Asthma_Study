import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

training_dataset = np.load('training_dataset.npy').T


range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#range_n_clusters = [2, 5]

inertia = {}
silhouette = {}

for n_clusters in range_n_clusters:
    print('number of clusters :{}'.format(n_clusters))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit(training_dataset)
    inertia[n_clusters] = kmeans.inertia_
    cluster_labels = kmeans.labels_


    silhouette_avg = silhouette_score(training_dataset, cluster_labels)
    vcr_avg = calinski_harabasz_score(training_dataset,cluster_labels)
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
plt.show()

plt.figure()
plt.plot(list(silhouette.keys()), list(silhouette.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette")
plt.show()