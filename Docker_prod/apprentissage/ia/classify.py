import pandas as pd
import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


classifier = None


def config (model_name) :  
    global classifier
    model_handle = "./graphs/"+model_name
    classifier = keras.models.load_model(model_handle, compile=False)
        
def classify(ecg_file):
    def preprocess_ecg(ecg):
        ecg_padded = pad_sequences(ecg, maxlen=100)
        return ecg_padded

    def load_ecg(file_url):
        data = pd.read_csv(file_url, sep='\t', header=None)
        sequences = data.iloc[:, 1:].values
        sequences = np.expand_dims(sequences, axis=2)
        labels = data.iloc[:, 0].values
        labels = np.where(labels==-1, 0, 1)
        labels = to_categorical(labels)
        return sequences, labels

    def build_result(probabilities):
    	return [{'class': int(np.argmax(prob)), 'probability': float(np.max(prob))} for prob in probabilities]


    ecg, labels = load_ecg(ecg_file)
    if classifier is None:
        raise Exception("No model has been loaded, call config first.")
    probabilities = classifier.predict(ecg)
    result = build_result(probabilities)
    return result

