
import numpy as np
from matplotlib import pyplot as plt

#We built this algorithm as well, and we thought we could include it as well because it portrays our efforts and the research we did to build our own bisecting k mean algorithm#

#-----------------------------------------------------------------------Converting data to points-------------------------------------------------------------------------#
def convert_to_2d_array(points):
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, -1)
    return points

#-----------------------------------------------------------------------Visualise Clusters function-------------------------------------------------------------------------#
def visualize_clusters(clusters):
    plt.figure()
    for cluster in clusters:
        points = convert_to_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

#-----------------------------------------------------------------------Calculating the SEE-------------------------------------------------------------------------#
def SSE(points):
    points = convert_to_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)

#-----------------------------------------------------------------------K Mean Algorithm-------------------------------------------------------------------------#
def kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    points = convert_to_2d_array(points)
    assert len(points) >= k, "Number of data points can't be less than k"

    best_sse = np.inf
    for ep in range(epochs):
        # Randomly initialize k centroids
        np.random.shuffle(points)
        centroids = points[0:k, :]

        last_sse = np.inf
        for it in range(max_iter):
            # Cluster assignment
            clusters = [None] * k
            for p in points:
                index = np.argmin(np.linalg.norm(centroids-p, 2, 1))
                if clusters[index] is None:
                    clusters[index] = np.expand_dims(p, 0)
                else:
                    clusters[index] = np.vstack((clusters[index], p))

            centroids = [np.mean(c, 0) for c in clusters]

            sse = np.sum([SSE(c) for c in clusters])
            gain = last_sse - sse
            if verbose:
                print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                       f'SSE: {sse:12.4f}, Gain: {gain:12.4f}'))
            if sse < best_sse:
                best_clusters, best_sse = clusters, sse
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = sse
    return best_clusters

#-----------------------------------------------------------------------Bisecting K Mean Algorithm-------------------------------------------------------------------------#
def bisecting_kmeans(points, k=2, epochs=10, max_iter=100, verbose=False):
    points = convert_to_2d_array(points)
    clusters = [points]
    while len(clusters) < k:
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        cluster = clusters.pop(max_sse_i)
        two_clusters = kmeans(
            cluster, k=2, epochs=epochs, max_iter=max_iter, verbose=verbose)
        clusters.extend(two_clusters)
    return clusters
