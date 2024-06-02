# -*- coding: utf-8 -*-
"""
Projet de semestre : Majeure Biotech & NUmérique 
Noémie Lacourt, Nour Kanaan et Chloé Gaugry
Juin 2024
"""

#☻Improtation des librairies
import pandas as pd
import numpy as np
import mne

# Téléchargement du modèle 
from xgboost import XGBRegressor
xgbModel = XGBRegressor()
xgbModel.load_model(r'C:\Users\lacou\xgbModel.json')

# Chemin vers votre fichier CSV
file = 'EEG_40yo.csv'

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


# Fonction de création des caractéristiques
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

# Création des caractéristiques
features = createFeatures(file)
# Ajout dans la dataframe
dfPred = pd.DataFrame([features], columns=featuresName)

#### Prédictions #####
X_test = dfPred.drop(columns=['age'])
y_test = dfPred['age']
y_pred = xgbModel.predict(X_test)
print("Actual age    : ",y_test[0])
print("Predicted age : ",y_pred[0])


#### Calcul de l'erreur moyenne absolue ####
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE) :", mae)