# Implémentation de DeepLabV3 avec TensorRT en C++

## Description du projet

Ce projet implémente l'inférence de l'algorithme DeepLabV3 pour la segmentation sémantique en utilisant TensorRT en C++/CUDA. L'objectif est de charger un modèle ONNX, effectuer l'inférence sur une image, réaliser le post-traitement et afficher le résultat de la segmentation.

## Prérequis

- CUDA
- TensorRT
- OpenCV
- CMake

## Structure du projet

Le projet est organisé comme suit :

project_root/
│\
├── models/\
│   └── export_model.py    # Script Python pour exporter le modèle DeepLabV3 en ONNX\
│\
├── include/\
│   ├── ImageProcessor.hpp # En-tête pour le traitement d'image\
│   └── TensorRTEngine.hpp # En-tête pour le moteur TensorRT\
│\
├── src/\
│   ├── ImageProcessor.cpp # Implémentation du traitement d'image\
│   ├── TensorRTEngine.cpp # Implémentation du moteur TensorRT\
│   └── main.cpp           # Point d'entrée du programme\
│\
├── CMakeLists.txt         # Fichier de configuration CMake\
│\
└── README.md  \

### Description des composants principaux :
1. export_model.py :\
Ce script Python est utilisé pour charger le modèle DeepLabV3 pré-entraîné depuis PyTorch et l'exporter au format ONNX. Pour l'utiliser, exécutez la commande :

`python3 models/export_model.py`

2.ImageProcessor :\
    Cette classe gère le prétraitement des images d'entrée et le post-traitement des résultats de segmentation. Elle inclut des fonctions pour :\

1.     Redimensionner et normaliser les images d'entrée
2.     Convertir les sorties du réseau en masques de segmentation
3.     Appliquer des palettes de couleurs pour la visualisation

3.TensorRTEngine :\
    Cette classe encapsule toutes les opérations liées à TensorRT. Elle est responsable de :

1.         Charger et optimiser le modèle ONNX
2.         Créer et gérer le moteur d'inférence TensorRT
3.         Exécuter l'inférence sur les images d'entrée
4.         Gérer les buffers d'entrée/sortie et les transferts de mémoire entre le CPU et le GPU

4. CMakeLists.txt :\
    Ce fichier configure le processus de compilation du projet, spécifiant les dépendances, les fichiers source à compiler, et les bibliothèques à lier.

Cette structure sépare clairement les différentes responsabilités du projet, facilitant la maintenance et l'extension future du code.


## Utilisation

Pour compiler et exécuter le projet, suivez ces étapes :\
1. Compilation du projet :
Depuis le répertoire racine du projet, exécutez les commandes suivantes :

```
mkdir build
cd build
cmake ..
make
```
Ces commandes créent un dossier de build, génèrent les fichiers de configuration avec CMake, et compilent le projet.


2. Exécution du programme :
Une fois la compilation terminée, vous pouvez exécuter le programme avec la commande suivante :

```
./main ../models/deeplabv3.onnx ../data/test_image.jpg ../models/deeplabv3.engine
```

Les arguments de la commande sont :

    ../models/deeplabv3.onnx : Chemin vers le modèle ONNX
    ../data/test_image.jpg : Chemin vers l'image de test
    ../models/deeplabv3.engine : Chemin où le moteur TensorRT optimisé sera sauvegardé

Résultats :
Après l'exécution, le programme générera deux fichiers dans le répertoire courant :

    segmentation_mask.png : Masque de segmentation coloré
    segmentation_mask_gray.png : Masque de segmentation en niveaux de gris


## Format de sortie du réseau


Le réseau DeepLabV3 produit un tenseur de sortie de forme [1, 21, H, W], où :
- 1 représente la taille du batch (une seule image traitée à la fois)
- 21 est le nombre de classes (20 classes d'objets + 1 classe de fond)
- H et W sont respectivement la hauteur et la largeur de l'image d'entrée

Chaque élément du tenseur représente un score pour une classe spécifique à une position donnée de l'image. Pour obtenir la segmentation finale, nous sélectionnons la classe avec le score le plus élevé pour chaque pixel.

Voici une visualisation du modèle obtenue avec l'outil Netron: (voir netron.png)

Cette visualisation montre clairement la couche de sortie du réseau et sa forme.

## Visualisation des résultats

Les résultats de la segmentation sont visualisés de la manière suivante :
1. Un masque en niveaux de gris est créé, où chaque niveau correspond à une classe.
2. Une palette de couleurs est appliquée à ce masque pour une meilleure distinction visuelle des classes.
3. Deux images sont générées :
   - `segmentation_mask.png` : Le masque de segmentation coloré
   - `segmentation_mask_gray.png` : Le masque de segmentation en niveaux de gris

Voici un exemple de résultat de segmentation : (voir dans le dossier data)

Cette visualisation permet une interprétation rapide et intuitive des résultats de segmentation produits par le réseau.

## Références

Le modèle utilisé dans ce projet est basé sur DeepLabV3, un réseau de pointe pour la segmentation sémantique. Pour plus de détails sur l'architecture et les principes derrière DeepLabV3, veuillez consulter le papier original :

    Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv preprint arXiv:1706.05587.
    https://arxiv.org/abs/1706.05587
