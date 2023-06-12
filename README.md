# Classification d'ECG 
> DA SILVA Jérémy & ROGUET William

## Objectifs
* Classer des ECG dit "normaux" et "anormaux" (infarctus).
* Mettre en place au minimum deux méthodes et les comparer (CNN et RNN obligatoires)

## Livrables
* Le code final
* Un rapport

## Dataset
* [Train](https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv)
* [Test](https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv)

---

## ToDo
* [x] Mise en place du CNN [aide](https://colab.research.google.com/drive/1mEuCNRJ91HrL2kypCSm0zlpm_QBrllo3?usp=sharing) ;
* [x] Mise en place du RNN [aide](https://colab.research.google.com/drive/11jd7HBVxnJtcL8-vbeaG5WJBWPIfZ8I7?usp=sharing);
* [x] Evaluer les perfomances des deux méthodes (temps de calcul, taux d'erreur, etc.) ;
* [x] Rendre le code clair et lisible ;
* [x] Faire le rapport ;
* [x] Corriger le rapport.

## Aborescence du projet

```bash
ECG_Classification
├── data # Contient les jeux de données
├── img # Image (et PDF) nous servant à comparer le CNN et le RNN
├── models # Enregistrement des modèles créés ainsi que leur résumé généré par Keras
│   ├── best-model_CNN.h5
│   ├── best-model_RNN.h5
│   ├── CNN.txt
│   └── RNN.txt
├── classification.py # contient les fonctions CNN et RNN
├── main.py # Script qui lance les fonctions CNN et RNN
├── utils.py # Contient les fonctions utilitaires pour charger et pré-traiter les données
├── requirements.txt # Contient les dépendances du projet
└── README.md
```

## Dépendances 

Merci de lancer la commande suivante dans un terminal :

`python -m pip install --upgrade pip ; pip install -r requirements.txt`

## Lancement du projet

Pour éxecuter le code, il vous suffit de lance le main.py.

Dans le fichier main.py vous pouvez modifier les paramètres suivants :

```python
# Paramètres CNN
# Ces paramètres possèdent des valeurs par défaut.
convol_padding: str ('same' or 'valid')
convol_stride: int
convol_kernel_size: int
convol_filters: int
convol_activation: str ('relu', 'sigmoid', 'softmax', 'tanh', 'linear', 'elu', 'selu', 'softplus', 'softsign')
pool_size: int
pool_stride: int
pool_padding: str ('same' or 'valid')
nb_classes: int
out_activation: str ('relu', 'sigmoid', 'softmax', 'tanh', 'linear', 'elu', 'selu', 'softplus', 'softsign')
nb_epochs: int
batch_size: int
```

```python
# Paramètres RNN
# Ces paramètres possèdent des valeurs par défaut.
nb_neurons: int
batch_size: int
epochs: int
loss: str fonction de coût ('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity')
optimizer: str fonction d'optimisation d'algorithme ('sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam')
monitor: str valeur à mesurer et contrôler ('val_loss', 'val_accuracy', 'loss', 'accuracy')
