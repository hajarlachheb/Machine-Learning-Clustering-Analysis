import numpy as np
import random
import load_datasets
from tqdm import tqdm

X,y = load_datasets.load_vowel()
df_array = np.array(X)

def findMinMax(ar):
    min_l = []
    max_l = []
    for i in range(len(ar[0])):
        min_l.append(ar[:][i].min())
        max_l.append(ar[:][i].max())

    return np.array(min_l), np.array(max_l)

def init_centroids(k,ar,min_values,max_values):

    dim = len(ar[0])

    centroids = []
    for _ in range(k):
        centroid = []
        for d in range(dim):
            centroid.append(random.uniform(min_values[d],max_values[d]))

        centroids.append(np.array(centroid))
    return centroids

#print(init_centroids(3,df_array,findMinMax(df_array)[0],findMinMax(df_array)[1]))
def euclidean_distance(a,b):

    dist = np.linalg.norm(a - b)
    return dist

def assign_cluster(seeds,ar):

    centroids = []


    for idx,x in enumerate(ar):
        error = []
        #calculate distance of each datapoint to the centroids
        for jdx,seed in enumerate(seeds):
            error.append(euclidean_distance(x,seed))
        # take the minimum distance and assign it as the cluster
        centroids.append(np.argmin(error))

    return np.array(centroids)

#print(assign_cluster(init_centroids(3,df_array,findMinMax(df_array)[0],findMinMax(df_array)[1]),df_array))


def update_centroids(k,centroids,assigned_to,X,algorithm='means'):
    mean_centroids = []
    output_centroids = []
    for cidx in range(k):
    #for assigned in list(set(assigned_to)):

        mean_centroid = []
        counter = 0
        for i,x in enumerate(X):
            # find the current cluster
            if assigned_to[i] == cidx:
                mean_centroid.append(x)
                counter += 1
        # if centroid has no membership add the old centroid.
        if not mean_centroid:
            mean_centroids.append(centroids[cidx])
        else:
            if algorithm == 'means':
                mean_centroids.append(np.array(mean_centroid).mean(axis=0))
                #mean_centroids.append(np.sum(mean_centroid)/counter)


            if algorithm == 'harmonic': #k-harmonic-means
                #avoid division by zero
                epsilon = 0.0001
                mean_centroids.append(counter/sum(1/(np.array(mean_centroid)+epsilon)))

    return mean_centroids


def kmeans(k,ar,algorithm):
    min_values, max_values = findMinMax(ar)
    centroids = init_centroids(k, ar,min_values, max_values)

    assigned_points = assign_cluster(centroids, ar)
    iterations = 0
    saved_centroids = []
    while(iterations < 50):
        previous_centroids = centroids
        prev_points = assigned_points
        assigned_points = assign_cluster(centroids, ar)
        centroids = update_centroids(k,previous_centroids,assigned_points,ar,algorithm)

        # compute error
        errors_centroids = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                errors_centroids.append(euclidean_distance(centroids[i], centroids[j]))



        deviations = [float(euclidean_distance(previous_centroids[i],centroids[i])) for i in range(len(previous_centroids))]

        # break condition if deviation to previous clusters is small

        iterations += 1
#        print('devs:' ,deviations)
#        print(sum(deviations))
#        print(iterations, 'errors: ',errors_centroids)
        if sum(deviations) < 1e-7:
#            print('stop')
            break

    #print("num_iter: ",iterations)
    return centroids


def kmain(k, ar, algorithm,n_init=10):
    output = []
    for _ in tqdm(range(n_init)):
        output.append(kmeans(k, ar, algorithm))
    return output


#n_iter = 6
#safe_centroids(n_iter,kmeans(2,df_array,'harmonic'))
#final_centroids = read_centroids(n_iter)

#clusters = kmeans(5,df_array,'harmonic')
#print('num_cluster: ',len(clusters))
#print(clusters)
#print(assign_cluster(final_centroids[0],df_array))

def performance_score(X,centroids):
    #inter cluster distance
    inter_dist = []
    for i,c in enumerate(centroids):
        for j in range(i+1,len(centroids)):
            inter_dist.append(euclidean_distance(c,centroids[j]))

    mean_inter = np.array(inter_dist).mean(axis=0)

    # intra cluster distance

    intra_dists = []

    for jdx, centroid in enumerate(centroids):

        dist = []
        # calculate distance of each datapoint to the centroids
        for idx, x in enumerate(X):
            dist.append(euclidean_distance(x, centroid))

        intra_dists.append(np.mean(dist))


    mean_intra = np.mean(intra_dists)

    # inter cluster distance should be big intra cluster distance should be small
    # given the final calculation a high score indicates a good score and a low score a bad score
    return mean_inter-mean_intra

#print(performance_score(df_array,clusters))


