import os
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_pen_based():
    # Load input dataset
    dataset = os.path.join('datasets', 'pen-based.arff')
    data = loadarff(dataset)
    df_data = pd.DataFrame(data[0])

    # Save classes of instances
    classes = df_data['a17'].astype(int)

    # Drop class column from dataframe
    df_data = df_data.drop(columns='a17')

    return df_data, classes


def load_vowel():
    # Load input dataset
    dataset = os.path.join('datasets', 'vowel.arff')
    data = loadarff(dataset)
    df_data = pd.DataFrame(data[0])

    # Preprocessing - encode categorical variables with label encoder
    cols = ['Sex', 'Train_or_Test', 'Speaker_Number', 'Class']
    le = LabelEncoder()
    df_data[cols] = df_data[cols].apply(le.fit_transform)

    # Save classes of instances
    classes = df_data['Class']

    # Drop class column from dataframe
    df_data = df_data.drop(columns='Class')

    return df_data, classes


def load_adult():
    # Load input dataset
    dataset = os.path.join('datasets', 'adult.arff')
    data = loadarff(dataset)
    df_data = pd.DataFrame(data[0])

    # Preprocessing
    df_data.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country',
                            'hours-per-week': 'hours per week', 'marital-status': 'marital'}, inplace=True)
    df_data['country'] = df_data['country'].replace('?', np.nan)
    df_data['workclass'] = df_data['workclass'].replace('?', np.nan)
    df_data['occupation'] = df_data['occupation'].replace('?', np.nan)

    df_data.dropna(how='any', inplace=True)

    # Encode the dataset with LabelEncoder
    label_encoder = LabelEncoder()

    catcolumns = list(df_data.select_dtypes(include=['object']).columns)
    for i in catcolumns:
        df_data[i] = label_encoder.fit_transform(df_data[i])

    # Scale the data with MinMaxScaler
    min_max_scaler = MinMaxScaler()

    scaled_df = pd.DataFrame()

    # Select all data columns except for the class one
    columnval = df_data.columns.values
    columnval = columnval[:-1]

    scaled_values = min_max_scaler.fit_transform(df_data[columnval])

    for i in range(len(columnval)):
        scaled_df[columnval[i]] = scaled_values[:, i]

    classes = df_data['class']

    return scaled_df, classes
