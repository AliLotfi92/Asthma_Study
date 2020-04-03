import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
import kmedoids
from sklearn.metrics.pairwise import pairwise_distances

num_principal = 4
training_dataset = np.load('training_dataset.npy').T
print('data shape:{}, number of pca components:{}'.format(np.shape(training_dataset), num_principal))

pca = PCA(n_components=num_principal)
principal_components = pca.fit_transform(training_dataset)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

PCA_components = pd.DataFrame(principal_components)

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1)
plt.show()

inertia = {}
silhouette = {}

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
D = pairwise_distances(PCA_components, metric='manhattan')

for n_clusters in range_n_clusters:
    print('number of clusters :{}'.format(n_clusters))

    M, C = kmedoids.kMedoids(D, n_clusters)
    labels = np.zeros(1173)

    for label in C:
        #print(label)
        for point_idx in C[label]:
            labels[point_idx] = label


    silhouette_avg = silhouette_score(PCA_components, labels)
    vcr_avg = calinski_harabasz_score(PCA_components,labels)
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