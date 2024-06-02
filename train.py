# -*- coding: utf-8 -*-
"""
Projet de semestre : Majeure Biotech & NUmérique 
Noémie Lacourt, Nour Kanaan et Chloé Gaugry
Juin 2024
"""

# Gestion des warnings
import warnings
warnings.filterwarnings("ignore")

#Récupération des données (fichier csv des EEG)
import os
dataTrain = os.listdir(r'C:\Users\lacou\Documents\ESME\4-Ingé 2\Projet\Dataset kaggle\data_eeg_age_v1\data2kaggle\train')
dataEval = os.listdir(r'C:\Users\lacou\Documents\ESME\4-Ingé 2\Projet\Dataset kaggle\data_eeg_age_v1\data2kaggle\eval')
data = dataTrain+dataEval

#### Création de la dataframe ####
# Nom des caractéristiques
featuresName = ['age', 
                'amplitudeDelta', 
                'amplitudeTheta', 
                'amplitudeAlpha', 
                'amplitudeBeta', 
                'amplitudeGamma', 
                'asymmetryDelta', 
                'asymmetryTheta', 
                'asymmetryAlpha', 
                'asymmetryBeta', 
                'asymmetryGamma',
                'entropyFP1', 
                'entropyFP2', 
                'entropyF3', 
                'entropyF4', 
                'entropyC3',
                'entropyC4',
                'entropyP3',
                'entropyP4',
                'Moyenne', 
                'Variance', 
                'Ecart-type', 
                'Skewness', 
                'Kurtosis']

# Création de la base de données vide
import pandas as pd
df = pd.DataFrame(index=range(len(dataTrain)),columns=featuresName)
df.head()


# Importation des librairies
import pandas as pd
import numpy as np
import mne


### Fonction de calcul des caractéristiques
def createFeatures(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        
    age_line = lines[0].strip() # Extraire l'âge de la première ligne
    age = int(age_line.split('=')[1].strip())

    # Extraire les noms des canaux de la deuxième ligne et supprimer les suffixes "-REF" et "EEG"
    channel_names_line = lines[1].strip()
    channel_names = channel_names_line.strip().split(',')
    channel_names = [name.lstrip('#').split('-')[0].strip().replace('EEG ', '') for name in channel_names]  # Supprimer "EEG" et les suffixes "-REF"
    
    # Lire les données EEG à partir de la troisième ligne
    eeg_data = []
    for line in lines[2:]:
        eeg_data.append(list(map(float, line.strip().split(','))))
    eeg_data = np.array(eeg_data).T  # Transposer pour obtenir (n_channels, n_times)

    
    # Calculer le premier quartile (Q1) et le troisième quartile (Q3) pour chaque canal
    Q1 = np.percentile(eeg_data, 25, axis=1)
    Q3 = np.percentile(eeg_data, 75, axis=1)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remplacer les valeurs aberrantes par la médiane de chaque canal
    medians = np.median(eeg_data, axis=1)[:, np.newaxis]  # Ajouter une dimension supplémentaire
    lower_bound = np.repeat(lower_bound[:, np.newaxis], eeg_data.shape[1], axis=1)
    upper_bound = np.repeat(upper_bound[:, np.newaxis], eeg_data.shape[1], axis=1)
    eeg_data_cleaned = np.where((eeg_data < lower_bound) | (eeg_data > upper_bound), medians, eeg_data)


    sfreq = 250  # Définir la fréquence d'échantillonnage : Par exemple, 250 Hz
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg') # Créer un objet Info pour stocker les métadonnées
    raw = mne.io.RawArray(eeg_data_cleaned, info) # Créer un RawArray avec les données et les métadonnées
    fmin, fmax = 1, 100  # Calculer la densité spectrale de puissance (PSD) avec la méthode Welch : Vous pouvez ajuster ces valeurs selon vos besoins
    n_fft = 500
    
    # Calculer la puissance spectrale
    exclude_channels = ['PHOTIC', 'IBI', 'BURSTS', 'SUPPR']
    include_channels = [i for i, ch in enumerate(channel_names) if ch not in exclude_channels]
    power, freqs = mne.time_frequency.psd_array_welch(raw._data[include_channels], sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)

    # Calculer la puissance moyenne dans différentes bandes de fréquences
    def bandpower(data, freqs, band): 
        band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_power = data[:, band_freqs].mean(axis=1)
        return band_power
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 35),
        "gamma": (35, 100) }
    
    # Amplitude des ondes (moyenne des puissances spectrales)
    band_powers = {band: bandpower(power, freqs, freq_range) for band, freq_range in bands.items()}
    amplitude_mean = {band: np.mean(powers) for band, powers in band_powers.items()} 
    
    # Asymétrie hémisphérique (différence entre hémisphère gauche et droit)
    left_channels = [ch for ch in channel_names if 'F3' in ch or 'C3' in ch or 'P3' in ch or 'O1' in ch or 'F7' in ch or 'T3' in ch or 'T5' in ch or 'A1' in ch]
    right_channels = [ch for ch in channel_names if 'F4' in ch or 'C4' in ch or 'P4' in ch or 'O2' in ch or 'F8' in ch or 'T4' in ch or 'T6' in ch or 'A2' in ch]
    left_indices = [channel_names.index(ch) for ch in left_channels]
    right_indices = [channel_names.index(ch) for ch in right_channels]
    asymmetry = {band: np.mean(band_powers[band][left_indices]) - np.mean(band_powers[band][right_indices]) for band in bands}
    
    # Entropie 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    eeg_data_normalized = scaler.fit_transform(eeg_data_cleaned)

    # Indices des canaux d'intérêt
    nameChannelsEntropy = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4','P3','P4']
    indices_channels_of_interest = [channel_names.index(ch) for ch in nameChannelsEntropy]
    eeg_data_filtered = eeg_data_normalized[indices_channels_of_interest]

    # Calculer l'entropie du signal EEG pour chaque canal
    from scipy.stats import entropy
    entropy_values = [entropy(channel_data) for channel_data in eeg_data_filtered]
    entropy = {channel: entropy_value for channel, entropy_value in zip(nameChannelsEntropy, entropy_values)}
    
    #Moyenne, Variance, Ecart-type, Skewness, Kurtosis
    eeg = eeg_data_cleaned[:-4]
    from scipy.stats import skew, kurtosis
    moy = np.mean(eeg, axis=1).mean()
    var = np.var(eeg, axis=1).mean()
    std = np.std(eeg, axis=1).mean()
    skew = skew(eeg, axis=1).mean()
    kurt = kurtosis(eeg, axis=1).mean()
        
    return age, amplitude_mean['delta'], amplitude_mean['theta'], amplitude_mean['alpha'], amplitude_mean['beta'], amplitude_mean['gamma'], asymmetry['delta'], asymmetry['theta'], asymmetry['alpha'], asymmetry['beta'], asymmetry['gamma'], entropy['FP1'], entropy['FP2'], entropy['F3'], entropy['F4'], entropy['C3'], entropy['C4'], entropy['P3'], entropy['P4'], moy, var, std, skew, kurt
  

### Fonction pour remplir la dataframe
def fillDataframe(df,i,features, featuresName):
    for j in range (len(features)):
        df.at[i, featuresName[j]] = features[j]

### Remplir la dataframe avec calcul des caractéristiques
for i,fName in enumerate(dataTrain):
    file = r'C:\Users\lacou\Documents\ESME\4-Ingé 2\Projet\Dataset kaggle\data_eeg_age_v1\data2kaggle\train/'+fName
    features = createFeatures(file)
    print(features)
    fillDataframe(df,i,features, featuresName)
    

# Les données étant de type object, on les converti en type float 
for feature in featuresName :
   df[feature] = df[feature].astype(float)


### Visualisation de valeurs aberrentes ####
#1ere methode
import matplotlib.pyplot as plt
import seaborn as sns

# Création de boxplots pour chaque caractéristique
for column in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.show()
       
#2eme methode
from scipy import stats
# Calculer le Z-Score pour chaque valeur de chaque colonne
z_scores = stats.zscore(df)
# Trouver les index des outliers
outliers = (abs(z_scores) > 3).any(axis=1)
outlier_indices = df[outliers].index
# Afficher les lignes considérées comme outliers
#print(df.loc[outlier_indices])
    
#3eme methode avec les quartiles
# Calculer l'IQR pour chaque colonne
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
# Trouver les index des outliers
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_indices = df[outliers].index
# Afficher les lignes considérées comme outliers
#print(df.loc[outlier_indices])

## Création d'une nouvelle base de données avec données corrigées
df_cleaned = df.copy()
Q1 = df_cleaned.quantile(0.25)
Q3 = df_cleaned.quantile(0.75)
IQR = Q3 - Q1
for column in df_cleaned.columns:
    median = df_cleaned[column].median()
    is_outlier = (df_cleaned[column] < (Q1[column] - 1.5 * IQR[column])) | (df_cleaned[column] > (Q3[column] + 1.5 * IQR[column]))
    df_cleaned.loc[is_outlier, column] = median
    
    
#### Séparation des données ####
X = df_cleaned.drop(columns=['age'])
y = df_cleaned['age']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

for column in X_train.columns:
    sns.histplot(df_cleaned[column], kde=True)
    plt.title(f'Distribution de {column}')
    plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribution de l\'âge')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.show()


## Modèle XGBoost avec GridSearchCV #####
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
# Création de l'instance et normalisation des données
xgb = XGBRegressor(random_state=1)
scaler = MinMaxScaler() 
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
from sklearn.model_selection import GridSearchCV
# Recherche des meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train_norm, y_train)
print("Meilleurs hyperparamètres :", grid_search.best_params_)
xgbModel = grid_search.best_estimator_

# Sauvegarde du modèle et du scaler
xgbModel.save_model('~/xgbModel.json')
import joblib
joblib.dump(scaler, 'scaler.pkl')


y_pred = xgbModel.predict(X_test_norm)
from sklearn.metrics import mean_squared_error as MSE
import math
mse=MSE(y_test,y_pred)
rmse=math.sqrt(mse)
print("RMSE :", rmse)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("R2 :", r2)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE) :", mae)



###### Graphique résultats #######
import matplotlib.pyplot as plt

# Création d'une dataframe à partir de y_test et y_pred
df_results = pd.DataFrame({'age': y_test, 'pred_age': y_pred})

# Fonction pour regrouper les âges par intervalles de 10 ans
def group_ages_by_10(age):
    return (age // 10) * 10

# Ajout d'une colonne 'age_group' à df_results
df_results['age_group'] = df_results['age'].apply(group_ages_by_10)

# Graphique MAE par groupe d'âge
mae_by_age_group = df_results.groupby('age_group').apply(lambda group: mean_absolute_error(group['age'], group['pred_age']))
plt.figure(figsize=(10, 6))
plt.bar(mae_by_age_group.index, mae_by_age_group.values, width=5)  # Largeur des barres définie sur 5
plt.xlabel('Groupe d\'âge')
plt.ylabel('MAE')
plt.title('MAE par groupe d\'âge (intervalles de 10 ans)')
plt.grid(axis='y')
plt.show()

# Graphique du nombre de personnes par groupe d'âge
df_ageGroupsTrain = pd.DataFrame({'age': y_train})
df_ageGroupsTrain['age_group'] = df_ageGroupsTrain['age'].apply(group_ages_by_10)
count_by_age_group = df_ageGroupsTrain['age_group'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(count_by_age_group.index, count_by_age_group.values, color='skyblue', width=5)
plt.xlabel('Groupe d\'âge')
plt.ylabel('Nombre de personnes')
plt.title('Nombre de personnes par groupe d\'âge (intervalles de 10 ans)')
plt.grid(axis='y')
plt.show()



### Modèle SVR ###
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100, 1000],
    'degree': np.arange(1, 6),
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 1, 10]
}
svr_cv = GridSearchCV(SVR(),param_grid, cv=5)
svr_cv.fit(X_train_std, y_train)
print(svr_cv.best_params_)
print(svr_cv.best_score_)
best_svr = svr_cv.best_estimator_
y_pred_svr = best_svr.predict(X_test_std)
mse=MSE(y_test,y_pred_svr)
rmse=math.sqrt(mse)
print("RMSE :", rmse)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred_svr)
print("R2 :", r2)


### Modèle Decision Tree Regressor ###
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=1)
param_grid = {
    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],  # Critère pour mesurer la qualité d'une division
    'splitter': ['best', 'random'],  # Stratégie utilisée pour choisir la division à chaque nœud
    'max_depth': [None, 10, 20, 30, 40, 50],  # Profondeur maximale de l'arbre
    'min_samples_split': [2, 10, 20],  # Nombre minimum d'échantillons requis pour diviser un nœud interne
    'min_samples_leaf': [1, 5, 10],  # Nombre minimum d'échantillons requis pour être à un nœud terminal
    'max_features': [None, 'auto', 'sqrt', 'log2'],  # Nombre de fonctionnalités à considérer lors de la recherche de la meilleure division
    'max_leaf_nodes': [None, 10, 20, 30, 40, 50]  # Nombre maximal de nœuds feuilles dans l'arbre
}
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train_norm, y_train)
y_pred_dt = dt.predict(X_test_norm)
mse=MSE(y_test,y_pred)
rmse=math.sqrt(mse)
print("RMSE :", rmse)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("R2 :", r2)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE) :", mae)

    
#### Graphique d'importances des caractéristiques  ###
# Obtenir les importances des caractéristiques
feature_importances = xgbModel.feature_importances_
features = X_train.columns

# Créer un DataFrame pour les importances des caractéristiques
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

import matplotlib.pyplot as plt
ff = xgbModel.feature_importances_
feat = pd.DataFrame(xgbModel.feature_importances_, index=X_train.columns, columns=["important"]).sort_values("important", ascending=False)
plt.barh(X_train.columns, xgbModel.feature_importances_)


