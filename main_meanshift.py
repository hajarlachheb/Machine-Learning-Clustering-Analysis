# Mean Shift Clustering algorithm
from sklearn.cluster import MeanShift
import load_datasets
import metrics


def run_meanshift(file):
    if file == 'pen-based':
        print("Pen-based dataset: \n")
        run_meanshift_penbased()

    elif file == 'vowel':
        print("Vowel dataset: \n")
        run_meanshift_vowel()

    elif file == 'adult':
        print("Adult dataset: \n")
        run_meanshift_adult()

    else:
        raise ValueError('Unknown dataset {}'.format(file))


def run_meanshift_vowel():
    X, y = load_datasets.load_vowel()
    results = []

    ms = MeanShift(max_iter=50)
    ms.fit(X.values)
    y_pred = ms.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


def run_meanshift_penbased():
    X, y = load_datasets.load_pen_based()
    results = []

    ms = MeanShift(max_iter=20)
    ms.fit(X.values)
    y_pred = ms.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


def run_meanshift_adult():
    X, y = load_datasets.load_adult()
    results = []

    ms = MeanShift(max_iter=50)
    ms.fit(X.values)
    y_pred = ms.labels_

    results.append((y, y_pred))
    metrics.run_metrics(y, y_pred)


run_meanshift('vowel')
run_meanshift('pen-based')
run_meanshift('adult')
