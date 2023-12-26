# Fuzzy C Means Clustering
import load_datasets
import metrics
from FuzzyCMeans import FuzzyCMeans


def run_fuzzycmeans(file):
    if file == 'pen-based':
        print('Pen-based dataset: \n')
        run_fuzzycmeans_penbased()

    elif file == 'vowel':
        print('Vowel dataset: \n')
        run_fuzzycmeans_vowel()

    elif file == 'adult':
        print('Adult dataset: \n')
        run_fuzzycmeans_adult()

    else:
        raise ValueError('Unknown dataset {}'.format(file))


def run_fuzzycmeans_vowel():
    X, y = load_datasets.load_vowel()
    results = []
    fuzzycmeans = FuzzyCMeans(c=11, m=2, n_init=10)
    y_pred = fuzzycmeans.fit_predict(X)
    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


def run_fuzzycmeans_penbased():
    X, y = load_datasets.load_pen_based()
    results = []
    fuzzycmeans = FuzzyCMeans(c=10, m=2, n_init=10)
    y_pred = fuzzycmeans.fit_predict(X)
    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


def run_fuzzycmeans_adult():
    X, y = load_datasets.load_adult()
    results = []
    fuzzycmeans = FuzzyCMeans(c=2, m=2, n_init=10)
    y_pred = fuzzycmeans.fit_predict(X)
    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


run_fuzzycmeans('vowel')
run_fuzzycmeans('pen-based')
run_fuzzycmeans('adult')
