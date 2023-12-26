# Agglomerative Clustering algorithm for Vowel and Pen-based data sets
from sklearn.cluster import AgglomerativeClustering
import load_datasets
import metrics



def run_agg_clust(file, metric, linkage):
    print("Agglomerative clustering with metric: " + str(metric) + " and linkage method: " + str(linkage))
    if file == 'pen-based':
        print("Pen-based data set: \n")
        run_aggclust_penbased(metric, linkage)

    elif file == 'vowel':
        print("Vowel data set: \n")
        run_aggclust_vowel(metric, linkage)

    # elif file == 'adult':
    #    print("Adult data set: \n")
    #    run_aggclust_adult(metric, linkage)

    else:
        raise ValueError('Unknown dataset {}'.format(file))


def run_aggclust_vowel(metric, linkage):
    results = []
    X, y = load_datasets.load_vowel()

    aggcluster = AgglomerativeClustering(n_clusters=11, affinity=metric, linkage=linkage)

    aggcluster.fit(X.values)
    y_pred = aggcluster.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


def run_aggclust_penbased(metric, linkage):
    results = []
    X, y = load_datasets.load_pen_based()

    aggcluster = AgglomerativeClustering(n_clusters=10, affinity=metric, linkage=linkage)

    aggcluster.fit(X.values)
    y_pred = aggcluster.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


run_agg_clust('vowel', 'euclidean', 'single')
run_agg_clust('vowel', 'euclidean', 'average')
run_agg_clust('vowel', 'euclidean', 'complete')
run_agg_clust('vowel', 'cosine', 'single')
run_agg_clust('vowel', 'cosine', 'average')
run_agg_clust('vowel', 'cosine', 'complete')

run_agg_clust('pen-based', 'euclidean', 'single')
run_agg_clust('pen-based', 'euclidean', 'average')
run_agg_clust('pen-based', 'euclidean', 'complete')
run_agg_clust('pen-based', 'cosine', 'single')
run_agg_clust('pen-based', 'cosine', 'average')
run_agg_clust('pen-based', 'cosine', 'complete')
