import pandas as pd
from numpy.random import RandomState
import numpy as np


# Check if X is a df and if so return only values (no labels)
def _get_values(X):
    return X.values if isinstance(X, pd.DataFrame) else X


# Fuzzy C-Means Class
class FuzzyCMeans:

    def __init__(self, c, m=2, threshold=1e-7, max_iter=100, n_init=10, random_state=None):
        self.c = c
        self.m = m
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_init = n_init
        self.V_ = None

        if random_state is not None:
            random_state = np.int64(random_state)
        self.random_state = RandomState(seed=random_state)

    def _compute_new_membership_values(self, X, V):
        """
        Compute new membership values (U matrix)
        :param X: Instances (data)
        :param V: Centroids matrix
        :return: New membership matrix
        """

        # Calculate distances from datapoints to centroids (euclidean distance)
        # Each row is going to be a different cluster and columns represent datapoints
        distances = np.empty((self.c, X.shape[0]))
        for i in range(self.c):
            diff = X - V[i]
            distances[i, :] = np.sqrt(np.einsum('ij,ij->i', diff, diff))  # euclidean distance with einstein sum

        # Initialise membership matrix (U). Set all values to 1 since is the min possible value
        # (All distances will be divided by themselves at some point, but we skip it by doing this)
        U = np.ones_like(distances)
        exp = 2 / (self.m - 1)

        # Compute new membership values
        for i in range(self.c):
            for j in range(self.c):
                if i != j:
                    U[i] += (distances[i] / distances[j])
        # Finish calculation of new membership values by raising to the -exp power and return new values
        return U ** -exp

    def _compute_new_centroids(self, X, U):
        """
        Compute new centroids matrix (V)
        :param X: Instances (data)
        :param U: Membership values (matrix U)
        :return: New centroids matrix (V)
        """
        # Create a CxN matrix where C = number of centroids and N = number of columns in X
        V = np.empty((self.c, X.shape[1]))

        # Raise to the power of m the membership matrix for later calculations
        U_exp_m = U ** self.m

        # Calculate new centroid and update V (matrix of centroids)
        for i in range(self.c):
            Um_trans = U_exp_m[i].reshape(-1, 1)  # transpose for calculus (matrix multiplication with X)
            V[i] = np.sum(Um_trans * X, axis=0) / np.sum(Um_trans)

        return V

    def fit(self, X):
        """
        Run the algorithm n_init times and keep
        the one with best cohesion
        """
        results = []
        for _ in range(self.n_init):
            res = self._fit(X)
            results.append(res)

        # Select the best result from the obtained
        best_res = np.argmin([res[4] for res in results])  # performance index is the 5th arg returned from _fit(X)
        (self.labels_, self.U_, self.V_, self.n_iter_, self.perfindex_) = results[best_res]
        return self

    def _fit(self, X):
        """
        Run main fuzzy c-means algorithm to classify the data into
        c labels or clusters
        """
        # Get values (check if X is df, if so select only X.values)
        X = _get_values(X)
        # Check if there are centroids pre-defined
        if self.V_ is None:
            # Choose c random initial centroids
            centroids_i = set()
            while len(centroids_i) != self.c:
                # Generate centroids randomly with a range big enough for all of them to be unique
                centroids_i = {self.random_state.randint(X.shape[0]) for _ in range(self.c)}

            V = X[list(centroids_i)] * 1.001  # Avoid divisions by 0
        else:
            V = self.V_

        # Initialise the current iteration variable with 0
        curr_iter = 0
        V_diff = self.threshold * 2

        # Run for max_iter times (100 by default) or until the difference between
        # all new and old centroids is less than threshold
        while curr_iter < self.max_iter and np.any(V_diff > self.threshold):
            # Compute new membership matrix (U)
            U = self._compute_new_membership_values(X, V)

            # Compute new centroids (V)
            V_old = V
            V = self._compute_new_centroids(X, U)

            # Calculate the difference between new and old centroids
            V_diff = np.linalg.norm(V - V_old, 2, axis=1)

            # Increase the value of current iteration variable
            curr_iter += 1

        # Select the maximum membership value for each datapoint to define the cluster label it belongs to
        labels = np.argmax(U, axis=0)

        # Calculate the performance index
        perfindex = self._performance_index(X, U, V)

        return labels, U, V, curr_iter, perfindex

    def _performance_index(self, X, U, V):
        # Initialise the inner errors array
        inner_errors = np.empty((self.c, X.shape[0]))

        # Calculate the inner errors for all clusters
        for i in range(self.c):
            # Sum of squared errors with Einstein sum
            diff = X - V[i]
            inner_errors[i, :] = np.einsum('ij,ij->i', diff, diff)

        global_mean = X.mean(axis=0)
        diff = V - global_mean
        outer_errors = np.einsum('ij,ij->i', diff, diff).reshape(-1, 1)

        # main formula
        perfindex = np.sum(U * (inner_errors - outer_errors))
        return perfindex

    def fit_predict(self, X):
        return self.fit(X).labels_
