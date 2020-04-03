import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score

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

for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit(PCA_components)
    inertia[n_clusters] = kmeans.inertia_
    cluster_labels = kmeans.labels_

    silhouette_avg = silhouette_score(PCA_components, cluster_labels)
    vcr_avg = calinski_harabasz_score(PCA_components, cluster_labels)
    silhouette[n_clusters] = silhouette_avg

    print("For n_clusters =", n_clusters,
          "The average of silhouette_score is :", silhouette_avg,
          "The average of VCR is:", vcr_avg)

    # sample_silhouette_values = silhouette_samples(training_dataset, cluster_labels)

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