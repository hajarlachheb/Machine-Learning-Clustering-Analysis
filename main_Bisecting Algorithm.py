import numpy as np
from load_datasets import load_vowel
from Algorithm_bmeans import bisectingKMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from metrics import run_metrics
from IPython import get_ipython
import load_datasets

#------------------------------------------------PEN BASED--------------------------------------------#
#Data Pen Based
points = np.array(load_datasets.load_pen_based.df_data_pen_based.values.tolist())
X, y = load_datasets.load_pen_based()

#Choosing the best  value -Elbow and Silouette Method-
from best_k_value import best_k_mean, choosing_k
best_k_mean(points)
choosing_k (points, 10)

#Testing with the first approach - Bisecting k means -
from Algorithm_bmeans2 import bisecting_kmeans, kmeans, visualize_clusters, convert_to_2d_array
clusters = bisecting_kmeans(points, k=10, epochs=10, max_iter=1000, verbose=False)
visualize_clusters(clusters)

#Second bisecting k mean method
c_list, c_info = bisectingKMeans(points, 10, 10)
visualize_clusters(c_list)
print(c_list)

#Metrics

run_metrics(y,c_info)


#------------------------------------------------Vowel--------------------------------------------#
#Data Vowel
points = np.array(load_vowel.df_data_vowel.values.tolist())

#Choosing the best  value -Elbow and Silouette Method-
from best_k_value import best_k_mean, choosing_k
best_k_mean(points)
choosing_k (points, 10)

#Testing with the first approach - Bisecting k means -
from Algorithm_bmeans2 import bisecting_kmeans, kmeans, visualize_clusters, convert_to_2d_array
clusters = bisecting_kmeans(points, k=11, epochs=10, max_iter=1000, verbose=False)
visualize_clusters(clusters)

#Second bisecting k mean method
c_list, c_info = bisectingKMeans(points, 10, 10)
visualize_clusters(c_list)
print(c_list)
print(c_info)

#Metrics

run_metrics(y,c_info)


#------------------------------------------------Adult--------------------------------------------#
#Data Vowel
points = np.array(load_adult.df_scaled.values.tolist())

#Choosing the best  value -Elbow and Silouette Method-
from best_k_value import best_k_mean, choosing_k
best_k_mean(points)
choosing_k (points, 10)

#Testing with the first approach - Bisecting k means -
from Algorithm_bmeans2 import bisecting_kmeans, kmeans, visualize_clusters, convert_to_2d_array
clusters = bisecting_kmeans(points, k=8, epochs=10, max_iter=1000, verbose=False)
visualize_clusters(clusters)

#Second bisecting k mean method
c_list, c_info = bisectingKMeans(points, 10, 10)
visualize_clusters(c_list)
print(c_list)
print(c_info)

#Metrics
run_metrics(y,c_info)