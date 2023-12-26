# Agglomerative Clustering algorithm for adult data set: NOT RECOMMENDED TO RUN DUE TO COMPUTATIONAL COST
from sklearn.cluster import AgglomerativeClustering
import load_datasets
import metrics


def run_aggclust_adult(metric, linkage):
    results = []
    X, y = load_datasets.load_adult()

    aggcluster = AgglomerativeClustering(n_clusters=2, affinity=metric, linkage=linkage)

    aggcluster.fit(X.values)
    y_pred = aggcluster.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


run_aggclust_adult('euclidean', 'single')
run_aggclust_adult('euclidean', 'average')
run_aggclust_adult('euclidean', 'complete')
run_aggclust_adult('cosine', 'single')
run_aggclust_adult('cosine', 'average')
run_aggclust_adult('cosine', 'complete')
