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

## Apprentissage auto-supervisé (créer les labels à partir des données / détecter des nouveaux comportements)

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
- Kimi-Audio : Audio brut resampling 16kHz (préféré)
- DinoV2 : Mel spectrogramme

### Métriques de comparaison des modèles (après avoir fait tourné même Random Forest sur embeddings)

- Accuracy
- F1 score macro
- Confusion Matrix
- ROC AUC (multi class)

## Points d'attention

- Déséqulibre des classes (pondération des classes, oversampling, downsampling)
- Peu de données (data augmentation)
- Techniques de DL très légèrement meilleur que ML sur l'accélérométrie (https://doi.org/10.48550/arXiv.2509.08606)

## Ligne de conduite

- pré processing si nécessaire sur audio brut
- Obtenir des embeddings avec chacun des cinq modèles via feature extractor
- Implémenter un random forest similaire à celui de Samuel (pour approcher ses métriques obtenues avec Dinov2)
- Entraîner le même random forest sur chacun des cinq jeux d'embeddings et calculer les métriques correspondantes
- Benchmark sur les métriques
- tester random forest avec des descripteurs manuels au lieu des embeddings