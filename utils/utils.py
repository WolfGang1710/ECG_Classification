import numpy as np
import pandas as pd

pathTrain = "data/ECG200_TRAIN.tsv"
pathTest = "data/ECG200_TEST.tsv"


def load_data():
    test = np.array(pd.read_csv(pathTest, delimiter='\t', header=None))
    train = np.array(pd.read_csv(pathTrain, delimiter='\t', header=None))

    x_train = train[:, 1:]
    y_train = train[:, 0]
    x_test = test[:, 1:]
    y_test = test[:, 0]

    return (x_train, y_train), (x_test, y_test)


# Créé une fonction pour normaliser les données d'un tabeau numpy
def normalize(data):
    mini = np.min(data)
    maxi = np.max(data)
    range_value = maxi - mini
    return (data - mini) / range_value


