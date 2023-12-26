# Metrics
from sklearn.metrics import adjusted_rand_score, homogeneity_score, accuracy_score, f1_score


def run_metrics(y, y_pred):
    # Adjusted Random Score
    ars = adjusted_rand_score(y, y_pred)

    # Homogeneity score
    hs = homogeneity_score(y, y_pred)

    # Accuracy score
    accsc = accuracy_score(y, y_pred)

    # F-Score
    fs = f1_score(y, y_pred, average='weighted')

    # Print metric results
    print("Results are: \n" +
          "Adjusted Random Score: " + str(ars) + "\n" +
          "Homogeneity score: " + str(hs) + "\n" +
          "Accuracy score: " + str(accsc) + "\n"
          "F-score: " + str(fs) + "\n"
          )
