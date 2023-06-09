"""
@authors : DA SILVA Jérémy & ROGUET William
"""
import os # Gestion de fichiers
import shutil # Suppression de dossiers
import matplotlib.pyplot as plt
from utils import * # Fonctions utilitaires
from classification import * # Fonctions de classification

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

loss_train_epochs, loss_val_epochs, acc_train_epochs, acc_val_epochs, model = cnn(x_train, y_train, x_test, y_test)

f = open("models/cnn.txt", "a")
_, acc = model.evaluate(x_train, y_train, verbose=False)
print("L'accuracy sur l'ensemble du train est :", acc)
f.write("L'accuracy sur l'ensemble du train est :"+str(acc)+'\n')

_, acc = model.evaluate(x_test, y_test, verbose=False)
f.write("L'accuracy sur l'ensemble du test est :"+str(acc)+'\n')
print("L'accuracy sur l'ensemble du test est :", acc)
f.close()

plt.figure()
plt.title("Loss en fonction du nombre d'epochs pour le CNN")
plt.plot(loss_train_epochs, color='blue', label='train_loss')
plt.plot(loss_val_epochs, color='red', label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('img/epochs-loss_CNN.pdf')
plt.savefig('img/epochs-loss_CNN.png')
plt.show()
plt.close()

plt.figure()
plt.title("Accuracy en fonction du nombre d'epochs pour le CNN")
plt.plot(acc_train_epochs, color='blue', label='train_acc')
plt.plot(acc_val_epochs, color='red', label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('img/epochs-acc_CNN.pdf')
plt.savefig('img/epochs-acc_CNN.png')
plt.show()
plt.close()

loss_train_epochs, loss_val_epochs, acc_train_epochs, acc_val_epochs, model = rnn(x_train, y_train, x_test, y_test)

f = open("models/rnn.txt", "a")
_, acc = model.evaluate(x_train, y_train, verbose=False)
print("L'accuracy sur l'ensemble du train est :", acc)
f.write("L'accuracy sur l'ensemble du train est :"+str(acc)+'\n')

_, acc = model.evaluate(x_test, y_test, verbose=False)
f.write("L'accuracy sur l'ensemble du test est :"+str(acc)+'\n')
print("L'accuracy sur l'ensemble du test est :", acc)
f.close()

plt.figure()
plt.title("Loss en fonction du nombre d'epochs pour le RNN")
plt.plot(loss_train_epochs, color='blue', label='train_loss')
plt.plot(loss_val_epochs, color='red', label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('img/epochs-loss_RNN.pdf')
plt.savefig('img/epochs-loss_RNN.png')
plt.show()
plt.close()

plt.figure()
plt.title("Accuracy en fonction du nombre d'epochs pour le RNN")
plt.plot(acc_train_epochs, color='blue', label='train_acc')
plt.plot(acc_val_epochs, color='red', label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('img/epochs-acc_RNN.pdf')
plt.savefig('img/epochs-acc_RNN.png')
plt.show()
plt.close()
