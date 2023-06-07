from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from utils.utils import load_data


def CNN(padding_conv='same', stride=1, kernel_size=3, filters=8, activation='relu', pool_size=2,
        stride_pool=2, padding_pool='same', learning_rate=0.01, metrics=['accuracy'],
        mini_batch_size=32, nb_epochs=100, percentage_of_train_as_validation=0.3):
    """
    Modèle CNN pour la classification d'ECG
    :param padding_conv: str 'same' ou 'valid'
    :param stride: int
    :param kernel_size: int
    :param filters: int
    :param activation: str fonction d'activation
    :param pool_size: int
    :param stride_pool: int
    :param padding_pool: str 'same' ou 'valid'
    :param learning_rate: float entre 0 et 1
    :param metrics: valeur à mesurer
    :param mini_batch_size: int
    :param nb_epochs: int nombre d'epoques
    :param percentage_of_train_as_validation: float entre 0 et 1 pour la validation
    :return: model
    """
    (x_train, y_train), (x_test, y_test) = load_data()

    # Normalisation des données
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print(x_train)
    print(x_train.shape)

    # Reshape des données
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape)

    # Créer la couche d'entrée qui a la même shape que celle d'une instance dans x_train
    inputs = keras.Input(x_train.shape[1:])
    # créer la couche convolutive 1D en lui spécifiant les hyper-paramètres et la lier à la couche d'entrée
    hidden_conv_layer_1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                                        padding=padding_conv, activation=activation)(inputs)
    # Lier un max pooling à la couche convolutive
    pooling_conv_layer_1 = layers.MaxPooling1D(pool_size=pool_size, strides=stride_pool,
                                               padding=padding_pool)(hidden_conv_layer_1)
    # Aplatir la sortie du pooling
    flatten_layer = layers.Flatten()(pooling_conv_layer_1)
    # Créer la couche de sortie
    outputs = layers.Dense(1, activation='softmax')(flatten_layer)

    # Créer le modèle
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN")

    # Afficher le résumé du modèle
    model.summary()

    # Choisir l'algorithme d'optimisation avec un learning rate de 0.01
    optimizer_algo = keras.optimizers.SGD(learning_rate=learning_rate)
    # Choisir la fonction de coût
    cost_function = keras.losses.binary_crossentropy
    # Compiler le modèle en lui indiquant qu'on veut mesurer aussi l'accuracy
    model.compile(loss=cost_function, optimizer=optimizer_algo, metrics=metrics)
    model_checkpoint = keras.callbacks.ModelCheckpoint('models/best-model_CNN.h5', monitor='val_loss',
                                                       save_best_only=True)

    # Entraîner le modèle
    history = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                        verbose=False, validation_split=percentage_of_train_as_validation,
                        callbacks=[model_checkpoint])

    # Tracer la variation du taux d'erreur sur le train et sur le validation set en fonction du nombre d'epoques
    history_dict = history.history
    loss_train_epochs = history_dict['loss']
    loss_val_epochs = history_dict['val_loss']

    plt.figure()
    plt.plot(loss_train_epochs, color='blue', label='train_loss')
    plt.plot(loss_val_epochs, color='red', label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Evolution de la fonction de coût en fonction du nombre d\'époques pour le CNN')
    plt.savefig('img/epoch-loss_CNN.pdf')
    plt.savefig('img/epoch-loss_CNN.png')
    plt.show()
    plt.close()

    model = keras.models.load_model('models/best-model_CNN.h5')

    loss, acc = model.evaluate(x_train, y_train, verbose=False)
    print("L'accuracy sur l'ensemble du train est:", acc)

    loss, acc = model.evaluate(x_test, y_test, verbose=False)
    print("L'accuracy sur l'ensemble du test est:", acc)

    return model
