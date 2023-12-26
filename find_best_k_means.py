import numpy as np
from matplotlib import pyplot as plt
import load_datasets

from metrics import run_metrics
from kmeans import kmain, performance_score, assign_cluster, findMinMax, init_centroids
from tqdm import tqdm

# Determine the best k value:
# Pen-based dataset:
k_val = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
k_val = [10,11]
X, y = load_datasets.load_pen_based()
score_pen_l = []
choose_pen_best = []
n_init=1
for k in k_val:

    centroids_pen = kmain(k=k, ar=np.array(X), algorithm='means', n_init=n_init)


    scores_pen = []
    for i in range(n_init):
        scores_pen.append(performance_score(np.array(X),centroids_pen[i]))

    print(np.max(scores_pen),np.argmax(scores_pen))
    score_pen_l.append(scores_pen)
    choose_pen_best.append(np.argmax(scores_pen))

k_scores = []
for idx,choice in enumerate(choose_pen_best):

    k_scores.append(score_pen_l[idx][choice])
print(k_scores)
print(np.argmax(k_scores))
print(score_pen_l)
print(choose_pen_best)
#plt.figure()
#plt.plot(k_val, score_pen_l)
#plt.title('Score vs k - Pen-based dataset (FCM)')
#plt.xlabel('k value')
#plt.ylabel('Score')
#plt.show()

# Vowel dataset:
k_val = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
k_val = [11, 12]
X, y = load_datasets.load_vowel()
score_vowel_l = []
choose_vowel_best = []

for k in k_val:

    centroids_vowel = kmain(k=k, ar=np.array(X), algorithm='means', n_init=n_init)


    scores_vowel = []
    for i in range(n_init):
        scores_vowel.append(performance_score(np.array(X),centroids_vowel[i]))

    print(np.max(scores_vowel),np.argmax(scores_vowel))
    score_vowel_l.append(scores_vowel)
    choose_vowel_best.append(np.argmax(scores_vowel))
print(choose_vowel_best)
#plt.figure()
#plt.plot(k_val, score_vowel_l)
#plt.title('Score vs k - Vowel dataset (FCM)')
#plt.xlabel('k value')
#plt.ylabel('Score')
#plt.show()

# Adult dataset:
k_val = [1, 2, 3, 4, 5]

k_val = [2, 3]
X, y = load_datasets.load_adult()
score_adult_l = []
choose_adult_best = []

for k in k_val:

    centroids_adult = kmain(k=k, ar=np.array(X), algorithm='means', n_init=n_init)


    scores_adult = []
    for i in range(n_init):
        scores_adult.append(performance_score(np.array(X),centroids_adult[i]))


    print(np.max(scores_adult),np.argmax(scores_adult))
    score_adult_l.append(scores_adult)

    choose_adult_best.append(np.argmax(scores_adult))

final_best_adult = []

for best in choose_adult_best:

    performance_score(np.array(X),centroids_adult[i])
print(choose_adult_best)
#plt.figure()
#plt.plot(k_val, score_adult_l)
#plt.title('Score vs k - Adult dataset (FCM)')
#plt.xlabel('k value')
#plt.ylabel('Score')
#plt.show()

all_centroids = [centroids_pen,centroids_vowel,centroids_adult]

k_scores_pen = [score_pen_l[idx][choice] for idx,choice in enumerate(choose_pen_best)]
k_scores_vowel = [score_vowel_l[idx][choice] for idx,choice in enumerate(choose_vowel_best)]
k_scores_adult = [score_adult_l[idx][choice] for idx,choice in enumerate(choose_adult_best)]

best_centroids = [centroids_pen[np.argmax(k_scores_pen)],
                  centroids_vowel[np.argmax(k_scores_vowel)],
                  centroids_adult[np.argmax(k_scores_adult)]]

# here the evaluation should take place however there is some last lines missing to complete it
for idx,centroids in enumerate(best_centroids):
    if idx == 0:
        X, y = load_datasets.load_pen_based()
    if idx == 1:
        X, y = load_datasets.load_vowel()
    if idx == 2:
        X, y = load_datasets.load_adult()
    y = np.array(y)
    y_pred = assign_cluster(centroids,np.array(X))
    print(y_pred)


    for i in range(len(y)):
        print(run_metrics(y[i],y_pred[i]))