import numpy as np
from matplotlib import pyplot as plt
import load_datasets
from FuzzyCMeans import FuzzyCMeans

# Determine the best k value:
# Pen-based dataset:
k_val = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
X, y = load_datasets.load_pen_based()
sse_l = []
sse = 0
for k in k_val:
    fuzzycmeans = FuzzyCMeans(c=k, m=2, n_init=10)
    self = fuzzycmeans.fit(X)
    sse = 0
    for i in range(k):
        datapoints = X[self.labels_ == i]
        centroid = self.V_[i]
        sse += np.sum((datapoints - centroid) ** 2)
    sse_l.append(np.sum(sse))
print(sse_l)
plt.figure()
plt.plot(k_val, sse_l)
plt.title('SSE vs k - Pen-based dataset (FCM)')
plt.xlabel('k value')
plt.ylabel('SSE')
plt.show()

# Vowel dataset:
k_val = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
X, y = load_datasets.load_vowel()
sse_l = []
sse = 0
for k in k_val:
    fuzzycmeans = FuzzyCMeans(c=k, m=2, n_init=10)
    self = fuzzycmeans.fit(X)
    sse = 0
    for i in range(k):
        datapoints = X[self.labels_ == i]
        centroid = self.V_[i]
        sse += np.sum((datapoints - centroid) ** 2)
    sse_l.append(np.sum(sse))
print(sse_l)
plt.figure()
plt.plot(k_val, sse_l)
plt.title('SSE vs k - Vowel dataset (FCM)')
plt.xlabel('k value')
plt.ylabel('SSE')
plt.show()

# Adult dataset:
k_val = [1, 2, 3, 4, 5]
X, y = load_datasets.load_adult()
sse_l = []
sse = 0
for k in k_val:
    fuzzycmeans = FuzzyCMeans(c=k, m=2, n_init=10)
    self = fuzzycmeans.fit(X)
    sse = 0
    for i in range(k):
        datapoints = X[self.labels_ == i]
        centroid = self.V_[i]
        sse += np.sum((datapoints - centroid) ** 2)
    sse_l.append(np.sum(sse))
print(sse_l)
plt.figure()
plt.plot(k_val, sse_l)
plt.title('SSE vs k - Adult dataset (FCM)')
plt.xlabel('k value')
plt.ylabel('SSE')
plt.show()
