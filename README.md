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

