"""
@authors : DA SILVA Jérémy & ROGUET William
"""
import os
import shutil
import matplotlib.pyplot as plt
from utils import *
from classification import *

print(f"==============================================\n"
      f"\t\tClassification d'ECG.\n"
      f"==============================================\n"
      f"\n"
      f"Programme Python réalisé par Da Silva Jérémy & Roguet William.")

print(f"==============================================\n"
      f"\t\tInitialisation.\n"
      f"==============================================\n")

folders = ['models', 'img']

for folder in folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Contenu du dossier '{folder}' supprime.")
    os.makedirs(folder)
    print(f"Dossier '{folder}' cree.")

print(f"Termine."
      f"==============================================\n")

pathTrain = "data/ECG200_TRAIN.tsv"
pathTest = "data/ECG200_TEST.tsv"

(x_train, y_train), (x_test, y_test) = load_data(pathTrain, pathTest)
(x_train, y_train), (x_test, y_test) = preprocessing(x_train, y_train, x_test, y_test)

loss_train_epochs, loss_val_epochs = cnn(x_train, y_train, x_test, y_test)

plt.figure()
plt.title("Loss en fonction du nombre d'epochs pour le CNN")
plt.plot(loss_train_epochs, color='blue', label='train_loss')
plt.plot(loss_val_epochs, color='red', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('img/epoch-loss_CNN.pdf')
plt.savefig('img/epoch-loss_CNN.png')
plt.show()
plt.close()

loss_train_epochs, loss_val_epochs = rnn(x_train, y_train, x_test, y_test)

plt.figure()
plt.title("Loss en fonction du nombre d'epochs pour le RNN")
plt.plot(loss_train_epochs, color='blue', label='train_loss')
plt.plot(loss_val_epochs, color='red', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('img/epoch-loss_RNN.pdf')
plt.savefig('img/epoch-loss_RNN.png')
plt.show()
plt.close()
