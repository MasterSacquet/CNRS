# Idées générales

## Apprentissage supervisé (exploitation des annotations)

### Machine learning

Random Forest ou Balanced Random Forest, SVM, Gradient Boosting sur des descripteurs audio et d'accéléromètre. (Pour les descripteurs : https://doi.org/10.3389/fevo.2018.00171)

Descripteurs accéléromètre (sur chaque axe (x,y,z) et sur la magnitude (sqrt(x^2+y^2+z^2))) :

- Domaine temporel :
mean, SD, variance, Pitch (ratio between X, Y and Z axes), Roll (angle between Y and Z axes), Overall dynamic body
acceleration (Sum of the dynamic
acceleration values for X,Y,
and Z axes), min, max, range, median, MAD, RMS, Energy, jerk, autocorrelation, signal magnitude area, entropy
- Domaine fréquentiel : 
energy per axes, dominant frequency, spectral entropy, spectral centroid, power spectral density
- Autre :
corrélation entre axe (correlation(x,y), correlation(x,z), correlation(y,z)), MFCC (mean, SD)

Descripteurs audio :

- Domaine temporel :
mean, SD, RMS, Zero crossing rate, skewness, kurtosis, peak amplitude, crest factor
- Domaine fréquentiel (après FFT):
Spectral centroid, spectral bandwidth, spectral roll-off, spectral flatness, spectral contrast, dominant frequency, energy per frequence bands
- Autre :
MFCC (mean, SD), Mel-spectrogram (mean, SD), LM spectrogram (mel logarithmique https://doi.org/10.3390/jimaging8040096)

-> Pas besoin de beaucoup de données annotées.

### Deep learning

CNN sur spectrogrammes et CNN (U-Net https://doi.org/10.1016/j.ecoinf.2026.103761) ou RNN (LSTM, GRU) sur accéléromètre puis fusion multimodale.

-> Besoin de beaucoup de données annotées.

## Apprentissage semi-supervisé (exploitation des annotations et des données non annotées)

## Apprentissage auto-supervisé (Détecter des changements dans le signal audio)

Noise contrastive estimation (https://www.biorxiv.org/content/10.1101/2022.10.12.511740v2) : détecter les changements dans le signal audio (sans passer par le spectrogramme), tâche prétexte de pseudo-labels générés automatiquement avec calcul de dissimilarité entre fenêtres consécutives et détection de pic dans la dissimilarité.

## Apprentissage non supervisé 

### Clustering (découverte de nouveaux comportements)

UMAP

k-means sur embeddings, HDBSCAN

## Transfer learning

- BEATs (auto-supervisé) + Attentive probing (https://doi.org/10.1016/j.ecoinf.2026.103765), (https://github.com/microsoft/unilm/tree/master/beats)
- VGG16 (CNN) + LM spectrogram audio (https://doi.org/10.3390/jimaging8040096)
- dasheng (https://github.com/XiaoMi/dasheng)
- Kimi-audio (https://github.com/MoonshotAI/Kimi-Audio)
- Dinov2 (https://github.com/facebookresearch/dinov2)

### Données audio nécéssaires à chaque modèle

- BEATs : Audio brut resampling 16kHz (préféré)
- VGG16 : Mel spectrogramme
- dasheng : Audio brut resampling 16kHz (obligatoire)
- DinoV2 : Mel spectrogramme

### Métriques de comparaison des modèles (après avoir fait tourné même Random Forest sur embeddings)

- Balanced Accuracy
- F1 score macro
- Confusion Matrix
- ROC AUC (multi class)
- Temps de calcul embeddings
- Temps d'entraînement Random Forest
- Dimension vecteur d'embeddings

## Points d'attention

- Déséqulibre des classes (pondération des classes, oversampling, downsampling)
- Peu de données (data augmentation)
- Techniques de DL très légèrement meilleur que ML sur l'accélérométrie (https://doi.org/10.48550/arXiv.2509.08606)

## Tâches

### 1) Benchmark embeddings transfer learning sur random forest 

- Calculer les embeddings des données audio (soit sur donnée brute, soit sur mel-spectrogramme) pour chacun des cinq modèles identifiés (BEATs, DinoV2, VGG16, Kimi-Audio et Dasheng)
- Entraînement sur chacun des cinq jeux de données ainsi qu'un jeu de donnée de descripteurs manuels d'un même Random Forest (n_estimators: 200, max_features: sqrt, criterion: gini, min_samples_leaf: 5)
- Benchmark sur les métriques identifiées : macro f1 score, balanced accuracy, Confusion Matrix, ROC AUC (multi class), Temps de calcul embeddings, Temps d'entraînement Random Forest, Dimension vecteur d'embeddings
- Utilisation de 15% de la base de donnée initiale (avec la même distribution de labels) (10 325 fichiers)
- Calcul des embeddings sur cette nouvelle base de donnée pour chacun des cinq modèles

#### Temps de calcul embedding, random forest, dimension, nombre de paramètres et taille du fichier d'embeddings (.parquet):

- DinoV2 Large (dinov2_vitl14) : 12 minutes 28 secondes (colab avec gpu) ; dimension 1024 ; random forest entrainement 13 mins 51 s ; 300 millions params ; Taille fichier embeddings : 60 Mo
- BEATs (BEATs_iter3_plus_AS2M) : 4 minutes 38 secondes (colab avec gpu) ; dimension 768 ; random forest entrainement 5 mins 22 s ; 90 millions params ; Taille fichier embeddings : 45 Mo
- Dasheng (base) : 3 minutes (colab avec gpu); dimension 768 ; random forest entrainement 9 mins 36 s ; 86 millions params ; Taille fichier embeddings : 45 Mo
- VGG16 : 30 mins 58 s (colab avec gpu) ; dimension 4096 ; Pas assez de RAM sur colab pour random forest (beaucoup trop gourmand) ; 68 millions params ; Taille fichier embeddings : 478 Mo
- EfficientNet (EfficientNet-B0) : 10 mins 44 s (colab avec gpu) ; dimension 1280 ; random forest entrainement 26 mins 05 s ; 5.27 millions params ; Taille fichier embeddings : 188 Mo
- Descripteurs manuels :

### 2) Détection automatique de comportements via SSL + Clustering

- Extraction d'embeddings sur base de données via soit modèle pré-entraîné (cf. benchmark précédent), soit modèle fait maison (cf. https://www.biorxiv.org/content/10.1101/2022.10.12.511740v2) -> représentation riche du signal audio
- Cluster sur embeddings afin de segmenter en plusieurs classes (nombre de classe à déterminer automatiquement)
- Associer clusters à un label via cluster labeling, majority vote

A terme :

- Entraînement transformer audio pour détecter ces classes