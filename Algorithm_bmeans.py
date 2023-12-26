from IPython import get_ipython
import numpy as np
from numpy.dual import svd
from scipy.sparse import csr_matrix
from sklearn import metrics

#----------------------------------------------------------------Random Centroids-------------------------------------------------------------------------#

# First we will be creating the set of random K centroids for each feature of our dataset
def randCent(dataSet, K):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((K, n)))
    for j in range(n):
        minValue = min(dataSet[:, j])
        maxValue = max(dataSet[:, j])
        rangeValues = float(maxValue - minValue)
        # We need to pay attention to the fact that the centroids stay within the range of data
        centroids[:, j] = minValue + rangeValues * np.random.rand(K, 1)
    return centroids

#------------------------------------------------Here we are going to measure the euclidean distance---------------------------------------------------------------#
def distanceMeasure(vecOne, vecTwo):
    return np.sqrt(np.sum(np.power(vecOne - vecTwo, 2)))

#----------------------------------------------------------------K Means Clustering-------------------------------------------------------------------------#

def kMeans(dataSet, K, distMethods=distanceMeasure, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, K)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -2
            for j in range(K):
                distJI = distMethods(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # update all the centroids by taking the np.mean value of relevant data
        for cent in range(K):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

#----------------------------------------------------------------Bisecting K Means-------------------------------------------------------------------------#
def bisectingKMeans(dataSet, K, numIterations):
    m, n = dataSet.shape
    clusterInformation = np.mat(np.zeros((m, 2)))
    centroidList = []
    minSSE = np.inf
    # First we need to find the best clusters, this is why we need to iterate the whole dataset
    for i in range(numIterations):
        centroid, clusterAssment = kMeans(dataSet, 2)
        SSE = np.sum(clusterAssment, axis=0)[0, 1]
        if SSE < minSSE:
            minSSE = SSE
            tempCentroid = centroid
            tempCluster = clusterAssment
    centroidList.append(tempCentroid[0].tolist()[0])
    centroidList.append(tempCentroid[1].tolist()[0])
    clusterInformation = tempCluster
    minSSE = np.inf

    # We then need to choose the cluster with the maximum SSE
    while len(centroidList) < K:
        maxIndex = -2
        maxSSE = -1
        for j in range(len(centroidList)):
            SSE = np.sum(clusterInformation[np.nonzero(clusterInformation[:, 0] == j)[0]])
            if SSE > maxSSE:
                maxIndex = j
                maxSSE = SSE

    # We need to choose the clusters with minimum total SSE so as to store them into our centroidList
        for k in range(numIterations):
            pointsInCluster = dataSet[np.nonzero(clusterInformation[:, 0] == maxIndex)[0]]
            centroid, clusterAssment = kMeans(pointsInCluster, 2)
            SSE = np.sum(clusterAssment[:, 1], axis=0)
            if SSE < minSSE:
                minSSE = SSE
                tempCentroid = centroid.copy()
                tempCluster = clusterAssment.copy()
    #Here, we update the index and it information
        tempCluster[np.nonzero(tempCluster[:, 0] == 1)[0], 0] = len(centroidList)
        tempCluster[np.nonzero(tempCluster[:, 0] == 0)[0], 0] = maxIndex
        clusterInformation[np.nonzero(clusterInformation[:, 0] == maxIndex)[0], :] = tempCluster
        centroidList[maxIndex] = tempCentroid[0].tolist()[0]
        centroidList.append(tempCentroid[1].tolist()[0])
    return centroidList, clusterInformation
