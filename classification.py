from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from tensorflow import keras
from numpy import sqrt


def cnn(x_train, y_train, x_test, y_test,
        convol_padding='same', convol_stride=1, convol_kernel_size=3,
        convol_filters=8, convol_activation='relu',
        pool_size=2, pool_stride=2, pool_padding='valid',
        nb_classes=2, out_activation='softmax'):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param convol_padding: str ('same' or 'valid')
    :param convol_stride: int
    :param convol_kernel_size: int
    :param convol_filters: int
    :param convol_activation: str ('relu', 'sigmoid', 'softmax', 'tanh', 'linear', 'elu', 'selu', 'softplus', 'softsign')
    :param pool_size: int
    :param pool_stride: int
    :param pool_padding: str ('same' or 'valid')
    :param nb_classes: int
    :param out_activation: str ('relu', 'sigmoid', 'softmax', 'tanh', 'linear', 'elu', 'selu', 'softplus', 'softsign')
    :return: loss_train_epochs, loss_val_epochs
    """
    input_shape = x_train.shape[1:]
    input_layer = keras.layers.Input(input_shape)
    hidden_conv_layer_1 = keras.layers.Conv1D(padding=convol_padding,
                                              strides=convol_stride,
                                              kernel_size=convol_kernel_size,
                                              filters=convol_filters,
                                              activation=convol_activation)(input_layer)
    pooling_conv_layer_1 = keras.layers.MaxPooling1D(pool_size=pool_size,
                                                     strides=pool_stride,
                                                     padding=pool_padding)(hidden_conv_layer_1)
    # Applattissement de la sortie du pooling
    flattened_layer_1 = keras.layers.Flatten()(pooling_conv_layer_1)

    output_layer = keras.layers.Dense(units=nb_classes, activation=out_activation)(flattened_layer_1)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    print("==============================================")
    print("Modele cree ! ")
    print("==============================================\n")
    temp = open("models/CNN.txt", "w")
    print(model.summary())
    model.summary(print_fn=lambda x: temp.write(x + '\n'))
    temp.close()

    learning_rate = 0.01
    optimizer_algo = keras.optimizers.SGD(learning_rate=learning_rate)
    cost_function = keras.losses.categorical_crossentropy
    model.compile(loss=cost_function, optimizer=optimizer_algo, metrics=['accuracy'])
    model_checkpoint = keras.callbacks.ModelCheckpoint('models/best-model_CNN.h5', monitor='val_loss',
                                                       save_best_only=True)
    mini_batch_size = 256
    nb_epochs = 100
    percentage_of_train_as_validation = 0.3
    print("Debut de l'entrainement du modele !")
    history = model.fit(x_train, y_train, batch_size=mini_batch_size,
                        epochs=nb_epochs, verbose=False,
                        validation_split=percentage_of_train_as_validation,
                        callbacks=[model_checkpoint])

    history_dict = history.history
    loss_train_epochs = history_dict['loss']
    loss_val_epochs = history_dict['val_loss']

    return loss_train_epochs, loss_val_epochs


def rnn(x_train, y_train, x_test, y_test,
        nb_neurons=4, batch_size=1, epochs=100,
        loss='mean_squared_error', optimizer='adam',
        monitor='val_loss'):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param nb_neurons: int
    :param batch_size: int
    :param epochs: int
    :param loss: str fonction de coût ('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity')
    :param optimizer: str fonction d'optimisation d'algorithme ('sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam')
    :param monitor: str valeur à mesurer et contrôler ('val_loss', 'val_accuracy', 'loss', 'accuracy')
    :return: loss_train_epochs, loss_val_epochs
    """
    # Définition du modèle RNN
    input_layer = keras.layers.Input(batch_shape=[batch_size, x_train.shape[1], x_train.shape[2]])
    hidden_layer_1 = keras.layers.SimpleRNN(units=nb_neurons, stateful=True, return_sequences=True)(input_layer)
    hidden_layer_2 = keras.layers.SimpleRNN(units=nb_neurons, stateful=True)(hidden_layer_1)
    output_layer = keras.layers.Dense(units=2)(hidden_layer_2)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    # Compilation du modèle
    model.compile(loss=loss, optimizer=optimizer)

    # Entraînement du modèle
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[
                            keras.callbacks.ModelCheckpoint('models/best-model_RNN.h5',
                                    monitor=monitor,
                                    save_best_only=True),
                            keras.callbacks.TensorBoard(log_dir='logs/RNN')
                        ]
                        )

    # Sauvegarde du résumé du modèle
    with open('models/RNN.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    history_dict = history.history
    loss_train_epochs = history_dict['loss']
    loss_val_epochs = history_dict['val_loss']

    return loss_train_epochs, loss_val_epochs
