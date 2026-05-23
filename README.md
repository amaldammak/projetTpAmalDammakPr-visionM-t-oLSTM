Projet  Prévision Météorologique avec LSTM

Amal Dammak — 2IDSD

*Présentation du projet

Dans ce projet, j’ai appliqué un réseau de neurones récurrents de type LSTM pour la prévision de séries temporelles météorologiques. L’objectif est de prédire la température (°C) à +24h à partir des observations passées.

Le modèle exploite 7 jours d’historique horaire (168 heures) afin d’estimer la température du jour suivant.

Le dataset utilisé est Jena Climate (2009–2016), issu d’une station météorologique en Allemagne. Il contient environ 420 000 observations et 14 variables climatiques enregistrées toutes les 10 minutes (température, pression, humidité, vent, etc.).

*Étapes réalisées
1. Exploration des données

Analyse des variables disponibles, étude statistique descriptive et vérification des valeurs manquantes. Compréhension des principales caractéristiques météorologiques du dataset.

2. Nettoyage et prétraitement
Correction des valeurs aberrantes de la vitesse du vent
Transformation de la direction du vent en composantes vectorielles (Wx, Wy)
Rééchantillonnage à une fréquence horaire pour réduire le bruit et accélérer l’apprentissage
Normalisation des données avec MinMaxScaler (appliquée uniquement sur les données d’entraînement pour éviter la fuite de données)
3. Feature engineering temporel

Ajout de variables cycliques (sin/cos) pour représenter :

l’heure de la journée
la saisonnalité annuelle

Cela permet au modèle de mieux capturer les phénomènes cycliques (ex : continuité entre 23h et 0h).

4. Préparation des séquences LSTM

Création de fenêtres temporelles :

X : 168 heures (7 jours) d’historique
y : température à +24h

Les données sont divisées chronologiquement :

70% entraînement
15% validation
15% test

sans mélange, afin de respecter l’ordre temporel.

5. Architecture du modèle

Le modèle est composé de :

BiLSTM (128 unités) → BatchNormalization → Dropout
LSTM (64 unités) avec mécanisme d’attention
Dense (32) → Dense (1)

Points clés :

Le BiLSTM capte les dépendances temporelles dans les deux directions
Le mécanisme d’attention met en avant les instants les plus importants
Les techniques de régularisation (Dropout, L2, BatchNorm) réduisent le surapprentissage

6. Entraînement
Fonction de perte : Huber, robuste aux valeurs extrêmes
Optimiseur : Adam (avec clipnorm pour stabiliser l’apprentissage)
Réduction automatique du learning rate (ReduceLROnPlateau)
EarlyStopping pour éviter le surapprentissage
Entraînement réalisé sur Google Colab (GPU T4)
7. Évaluation et résultats

Le modèle est évalué sur le jeu de test à l’aide de plusieurs métriques :

MAE : erreur moyenne en °C
RMSE : pénalise fortement les grandes erreurs
MAPE : erreur en pourcentage
R² : capacité du modèle à expliquer la variance
Comparaison avec un modèle baseline

Le modèle LSTM est comparé à un baseline de persistance, qui consiste à prédire que la température dans 24h sera identique à celle observée 24h auparavant.

Modèle	MAE	RMSE
Persistence	2.502°C	3.253°C
LSTM + Attention	2.458°C	3.089°C

Amélioration :

MAE : ↓ 1.8%
RMSE : ↓ 5.0%
Analyse des résultats

Le modèle LSTM dépasse le baseline sur toutes les métriques. L’amélioration du RMSE montre qu’il réduit particulièrement les erreurs importantes, ce qui est crucial pour la prévision météorologique.

L’analyse des résidus montre une distribution centrée autour de zéro, ce qui indique l’absence de biais systématique.

Le mécanisme d’attention améliore la performance en permettant au modèle de se concentrer sur les périodes les plus pertinentes de la séquence (notamment les variations récentes et les cycles journaliers).

*Technologies utilisées

Python 3.10
TensorFlow / Keras
NumPy, Pandas, Scikit-learn
Matplotlib, Seaborn
Google Colab (GPU T4)
