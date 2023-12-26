import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

#-----------------------------------------------------------------------Elbow Method-------------------------------------------------------------------------#
def best_k_mean(df) :
    inertia = []
    possible_K_values = [i for i in range(2, 40)]
    for each_value in possible_K_values:
        model = KMeans(n_clusters=each_value)
        model.fit(df)
        inertia.append(model.inertia_)
    plt.plot(possible_K_values, inertia)
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
#-----------------------------------------------------------------------Silhouette Method-------------------------------------------------------------------------#
def choosing_k (df,k) :
    bad_k_values = {}
    possible_K_values = [i for i in range(k, 30)]
    for each_value in possible_K_values:
        model = KMeans(n_clusters=each_value, init='k-means++', random_state=32)
        model.fit(df)
        silhouette_score_individual = silhouette_samples(df, model.predict(df))
        for each_silhouette in silhouette_score_individual:
            if each_silhouette < 0:
                if each_value not in bad_k_values:
                    bad_k_values[each_value] = 1
                else:
                    bad_k_values[each_value] += 1
    for key, val in bad_k_values.items():
        print(f' This Many Clusters: {key} | Number of Negative Values: {val}')
