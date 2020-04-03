import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
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
eps_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.1, 6.2, 6.8, 7]
min_samples_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


for eps in eps_range:
    for min_sample in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(PCA_components)
        cluster_labels = dbscan.labels_
        n_clusters = len(set(cluster_labels))
        print('number of clusters :{}'.format(n_clusters))
        if n_clusters > 1:
            silhouette_avg = silhouette_score(PCA_components, cluster_labels)
            vcr_av = calinski_harabasz_score(PCA_components, cluster_labels)
            print("For n_clusters: {}, the average of VCR : {}, the average of sitlhouette: {}, eps: {}, min_samples: {}".format(n_clusters, vcr_av, silhouette_avg, eps, min_sample))
