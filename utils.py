import numpy as np
import pandas as pd
from tensorflow import keras


def load_data(pathTrain, pathTest):
    """
    Charge les donnees d'entrainement et de test.
    :param pathTrain: str chemin vers le fichier d'entrainement
    :param pathTest: str chemin vers le fichier de test
    :return: (x_train, y_train), (x_test, y_test) : tuple de tuples contenant les donnees d'entrainement et de test
    """
    print(f"Chargement des donnees d'entrainement et de test.")
    test = np.array(pd.read_csv(pathTest, delimiter='\t', header=None))
    train = np.array(pd.read_csv(pathTrain, delimiter='\t', header=None))
    print(f"Termine."
          f"=========================================\n")

    x_train = train[:, 1:]
    y_train = train[:, 0]
    x_test = test[:, 1:]
    y_test = test[:, 0]

    return (x_train, y_train), (x_test, y_test)


def preprocessing(x_train, y_train, x_test, y_test):
    """
    Pretraite les donnees d'entrainement et de test.
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: (x_train, y_train), (x_test, y_test) : tuple de tuples contenant les donnees d'entrainement et de test
    """
    print(f"Preprocessing des donnees.")

    y_train = (y_train + 1) / 2
    y_test = (y_test + 1) / 2

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print(f"Termine.")
    print(f"Les donnees sont désormais normalisees et redimensionnees."
          f"=========================================\n")
    return (x_train, y_train), (x_test, y_test)
