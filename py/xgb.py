import sqlite3
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import gc
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance en km entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Version vectorisée de haversine pour numpy arrays."""
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcule l'azimut (bearing) entre deux points GPS en degrés."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(y, x)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Version vectorisée du calcul d'azimut."""
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing



def von_mises_weight(wind_dir, direction_to_emissaire, wind_dir_min, wind_dir_max):
    """
    Calcule un poids basé sur l'alignement entre le vent et la direction vers l'émissaire.
    Plus le vent pointe vers l'émissaire, plus le poids est élevé.
    """
    wind_rad = np.radians(wind_dir)
    target_rad = np.radians(direction_to_emissaire)
    
    amplitude = (wind_dir_max - wind_dir_min) % 360
    amplitude = np.where(amplitude > 180, 360 - amplitude, amplitude)
    
    kappa = 5 * (1 - amplitude / 180)
    kappa = np.maximum(kappa, 0.5)
    
    weight = np.exp(kappa * np.cos(wind_rad - target_rad))
    
    return weight

def von_mises_weight_coeff(wind_dir, direction_to_emissaire, wind_dir_min, wind_dir_max):
    wind_rad = np.radians(wind_dir)
    target_rad = np.radians(direction_to_emissaire)

    amplitude = (wind_dir_max - wind_dir_min) % 360
    amplitude = np.where(amplitude > 180, 360 - amplitude, amplitude)

    kappa = 5 * (1 - amplitude / 180)
    kappa = np.maximum(kappa, 0.5)

    # Normalisation pour que le poids soit entre 0 et 1
    weight = np.exp(kappa * np.cos(wind_rad - target_rad))
    weight = weight / (np.exp(kappa) + np.exp(-kappa))  # Normalisation par la somme des extrêmes
    return weight





def angular_difference(angle1, angle2):
    """Calcule la différence angulaire minimale entre deux angles."""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def calculate_certainty_for_emissaire(emissaire, all_emissaires, weather_point, wind_tolerance=30):
    """
    Calcule la certitude d'un émissaire basée sur :
    1. Absence d'autres émissaires dans le cône de vent
    2. Distance de l'émissaire le plus proche dans le cône
    3. Cohérence de la direction du vent avec la position de l'émissaire
    """
    emissaire_lat, emissaire_lon = emissaire['lat'], emissaire['lon']
    weather_lat, weather_lon = weather_point['latitude'], weather_point['longitude']
    wind_direction_10m = weather_point.get('wind_dir_10m', None)
    wind_direction_10mMin = weather_point.get('wind_dir_10mMin', None)
    wind_direction_10mMax = weather_point.get('wind_dir_10mMax', None)
    
    if pd.isna(wind_direction_10m):
        return 0.0
    
    # Direction vers l'émissaire depuis le point météo
    bearing_to_emissaire = calculate_bearing(weather_lat, weather_lon, emissaire_lat, emissaire_lon)


    ##N2GG suppr wind_aligment =angular_difference()
    weight = von_mises_weight_coeff(wind_direction_10m, bearing_to_emissaire, wind_direction_10mMin, wind_direction_10mMax)

    
    # Calculer la distance vers cet émissaire
    # Chercher d'autres émissaires dans le cône de vent
    competing_emissaires = []
    
    for idx, other_emissaire in all_emissaires.iterrows():
        if other_emissaire['id'] == emissaire['id']:
            continue
            
        other_lat, other_lon = other_emissaire['lat'], other_emissaire['lon']
        bearing_to_other = calculate_bearing(weather_lat, weather_lon, other_lat, other_lon)
        
        # Vérifier si l'autre émissaire est aussi dans le cône de vent
        other_wind_alignment = angular_difference(wind_direction_10m, bearing_to_other)
        other_wind_weight  = von_mises_weight_coeff(wind_direction_10m, bearing_to_other, wind_direction_10mMin, wind_direction_10mMax)
        
        if other_wind_weight <= wind_tolerance:
            distance_to_other = haversine(weather_lon, weather_lat, other_lon, other_lat)
            competing_emissaires.append({
                'id': other_emissaire['id'],
                'distance': distance_to_other,
                'bearing': bearing_to_other,
                'wind_weight': other_wind_weight
            })
    
    # Calcul de la certitude
    certainty = 1.0
    
    # Réduction basée sur l'alignement du vent (plus c'est aligné, mieux c'est)
    certainty *= weight
    
    # Réduction basée sur la présence d'émissaires concurrents
    if competing_emissaires:
        # Trouver le concurrent le plus lourd
        closest_competitor = max(competing_emissaires, key=lambda x: x['wind_weight'])
        
        if closest_competitor['wind_weight'] > weight:
            certainty *= np.exp(-0.1 * closest_competitor['wind_weight'])

    else:
        certainty *= 1.2  # Bonus de 20%
    
    # Normaliser entre 0 et 1
    certainty = max(0.0, min(1.0, certainty))
    
    return certainty

def load_data_and_calculate_certainties(weather_db_path, emissaire_db_path):
    """
    Charge les données et calcule les certitudes pour chaque combinaison météo-émissaire.
    """
    # Connexions aux bases
    weather_conn = sqlite3.connect(weather_db_path)
    emissaire_conn = sqlite3.connect(emissaire_db_path)
    
    # Charger tous les émissaires
    emissaire_query = "SELECT id, lat, lon FROM Emissaire WHERE type_='umapcrehum1'"
    emissaires = pd.read_sql_query(emissaire_query, emissaire_conn)
    print(f"Émissaires chargés: {len(emissaires)}")
    
    # Charger un échantillon de données météo pour le calcul des certitudes
    weather_query = """
    SELECT rowid, latitude, longitude, wind_dir_10m, wind_speed_10m,
           wind_dir_100m, wind_dir_10mMin, wind_dir_100mMin,
           wind_dir_10mMax, wind_dir_100mMax, temperature_2m, temperature_2m_min,
           temperature_2m_max, relative_humidity_2m, relative_humidity_2m_min,
           relative_humidity_2m_max, dewpoint_2m, dewpoint_2m_min, dewpoint_2m_max,
           rain, rain_min, rain_max, surface_pressure, surface_pressure_min,
           surface_pressure_max, et0_fao_evapotranspiration, et0_fao_evapotranspiration_min,
           et0_fao_evapotranspiration_max, vapour_pressure_deficit, vapour_pressure_deficit_min,
           vapour_pressure_deficit_max, wind_speed_10m_min,
           wind_speed_10m_max, wind_speed_100m, wind_speed_100m_min,
           wind_speed_100m_max, wind_gusts_10m, wind_gusts_10m_min,
           wind_gusts_10m_max, jourferieroudimanche
    FROM weather_data
    WHERE deltadaydeath = 0 
    AND wind_dir_10m IS NOT NULL 
    AND wind_speed_10m > 0.5
    ORDER BY RANDOM() 
    LIMIT 50000
    """
    weather_data = pd.read_sql_query(weather_query, weather_conn)
    print(f"Points météo chargés: {len(weather_data)}")
    
    weather_conn.close()
    emissaire_conn.close()
    
    return weather_data, emissaires

def create_training_dataset_with_certainties(weather_data, emissaires, certainty_threshold=0.3):
    """
    Crée le dataset d'entraînement avec calcul de certitudes pour chaque point météo.
    """
    training_data = []
    print("Calcul des certitudes et création du dataset...")
    
    # Traitement par chunks pour la mémoire
    chunk_size = 1000
    processed = 0
    
    for i in range(0, len(weather_data), chunk_size):
        chunk = weather_data.iloc[i:i+chunk_size]
        
        for weather_idx, weather_point in chunk.iterrows():
            if processed % 5000 == 0:
                print(f"  Traité: {processed}/{len(weather_data)} points météo...")
            
            # Calculer les distances vers tous les émissaires
            distances = haversine_vectorized(
                weather_point['longitude'], weather_point['latitude'],
                emissaires['lon'].values, emissaires['lat'].values
            )
            
            # Filtrer les émissaires dans un rayon raisonnable (1-200km)
            valid_mask = (distances >= 1.0) & (distances <= 200.0)
            valid_emissaires_idx = np.where(valid_mask)[0]
            
            for emissaire_idx in valid_emissaires_idx:
                emissaire = emissaires.iloc[emissaire_idx]
                distance = distances[emissaire_idx]
                
                # Calculer la certitude pour cette combinaison
                certainty = calculate_certainty_for_emissaire(
                    emissaire, emissaires, weather_point
                )
                
                # Ne garder que les cas avec certitude suffisante
                #N2GG TODO REV TODO : Les emiss semblent nombreusement gardés pour un Lieu ?!
                if certainty >= certainty_threshold:
                    training_row = weather_point.copy()
                    training_row['target_distance'] = distance
                    #training_row['emissaire_id'] = emissaire['id']
                    training_row['certainty'] = certainty
                    #training_row['emissaire_lat'] = emissaire['lat']
                    #training_row['emissaire_lon'] = emissaire['lon']
                    
                    # Ajouter des features dérivées de position
                    #N2GG ANTIWEIGHT

                    #N2GG weight dejà inclus dans coeff
                    #weight = von_mises_weight_coeff(weather_point['wind_dir_10m'], bearing_to_emissaire, weather_point['wind_dir_10mMin'], weather_point['wind_dir_10mMax'])
                    #training_row['weight'] = weight

                    bearing = calculate_bearing(
                        weather_point['latitude'], weather_point['longitude'],
                        emissaire['lat'], emissaire['lon']
                    )
                    training_row['bearing_to_emissaire'] = bearing
                    
                    #Probablement peu pertinent car angle bof
                    #if not pd.isna(weather_point['wind_dir_10m']):
                    #    training_row['wind_emissaire_alignment'] = angular_difference(
                    #        weather_point['wind_dir_10m'], bearing
                    #    )
                    #else:
                    #    training_row['wind_emissaire_alignment'] = np.nan
                    
                    training_data.append(training_row)
            
            processed += 1
    
    if not training_data:
        print("❌ Aucune donnée d'entraînement avec certitude suffisante!")
        return pd.DataFrame()
    
    training_df = pd.DataFrame(training_data)
    print(f"Dataset créé: {len(training_df)} échantillons")
    print(f"Certitude moyenne: {training_df['certainty'].mean():.3f}")
    print(f"Distance moyenne: {training_df['target_distance'].mean():.1f} km")
    
    return training_df

def prepare_features_for_training(df):
    """
    Prépare les features pour l'entraînement XGBoost.
    """
    weather_features = [
        #'wind_dir_10m', 'wind_dir_100m', 'wind_dir_10mMin', 'wind_dir_100mMin',
        #'wind_dir_10mMax', 'wind_dir_100mMax',
        'temperature_2m', 'temperature_2m_min',
        'temperature_2m_max', 'relative_humidity_2m', 'relative_humidity_2m_min',
        'relative_humidity_2m_max', 'dewpoint_2m', 'dewpoint_2m_min', 'dewpoint_2m_max',
        'rain', 'rain_min', 'rain_max', 'surface_pressure', 'surface_pressure_min',
        'surface_pressure_max', 'et0_fao_evapotranspiration', 'et0_fao_evapotranspiration_min',
        'et0_fao_evapotranspiration_max', 'vapour_pressure_deficit', 'vapour_pressure_deficit_min',
        'vapour_pressure_deficit_max', 'wind_speed_10m', 'wind_speed_10m_min',
        'wind_speed_10m_max', 'wind_speed_100m', 'wind_speed_100m_min',
        'wind_speed_100m_max', 'wind_gusts_10m', 'wind_gusts_10m_min',
        'wind_gusts_10m_max', 'jourferieroudimanche'
    ]
    
    # Features dérivées importantes
    derived_features = ['bearing_to_emissaire', #'wind_emissaire_alignment'
                        ]
    
    df_features = df.copy()
    
    # Features d'amplitude et stabilité
    if 'wind_dir_10mMin' in df_features.columns and 'wind_dir_10mMax' in df_features.columns:
        df_features['wind_dir_amplitude'] = np.where(
            (df_features['wind_dir_10mMax'] - df_features['wind_dir_10mMin']) > 180,
            360 - (df_features['wind_dir_10mMax'] - df_features['wind_dir_10mMin']),
            df_features['wind_dir_10mMax'] - df_features['wind_dir_10mMin']
        )
        weather_features.append('wind_dir_amplitude')
    
    if 'wind_speed_10m_min' in df_features.columns and 'wind_speed_10m_max' in df_features.columns:
        df_features['wind_speed_amplitude'] = df_features['wind_speed_10m_max'] - df_features['wind_speed_10m_min']
        weather_features.append('wind_speed_amplitude')
    
    if 'temperature_2m_min' in df_features.columns and 'temperature_2m_max' in df_features.columns:
        df_features['temperature_amplitude'] = df_features['temperature_2m_max'] - df_features['temperature_2m_min']
        weather_features.append('temperature_amplitude')
    
    final_features = weather_features + derived_features
    return df_features[final_features], final_features

def train_distance_predictor_with_certainty(training_df):
    """
    Entraîne XGBoost avec pondération par certitude.
    """
    print("Préparation des features...")
    X, feature_names = prepare_features_for_training(training_df)
    y = training_df['target_distance']
    weights = training_df['certainty']  # Pondération par certitude
    
    # Nettoyage
    mask = ~(X.isnull().any(axis=1) | y.isnull() | weights.isnull())
    X = X[mask]
    y = y[mask]
    weights = weights[mask]
    
    print(f"Données nettoyées: {len(X)} échantillons")
    print(f"Features utilisées: {len(feature_names)}")
    
    # Division stratifiée
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    print("Entraînement XGBoost avec pondération par certitude...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=10,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        min_child_weight=2,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement avec pondération
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    
    # Métriques pondérées
    weighted_mse = np.average((y_test - y_pred)**2, weights=w_test)
    weighted_mae = np.average(np.abs(y_test - y_pred), weights=w_test)
    
    # Métriques standards
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Weighted MSE: {weighted_mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Weighted MAE: {weighted_mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, feature_names, X_test, y_test, y_pred, w_test

def predict_for_all_weather_points(model, feature_names, weather_db_path, emissaire_db_path):
    """
    Applique le modèle à tous les points météo.
    """
    print("Prédiction pour tous les points météo...")
    
    conn = sqlite3.connect(weather_db_path)
    

    emissaire_conn = sqlite3.connect(emissaire_db_path)
    altitude_query = "SELECT AVG(altitude) FROM Emissaire WHERE type_='umapcrehum1'"
    emissaire_altitude = pd.read_sql_query(altitude_query, emissaire_conn).iloc[0, 0]
    emissaire_conn.close()
    print(f"Altitude moyenne des émissaires : {emissaire_altitude:.1f} m")

    
    # Charger tous les points météo
    query = """
    SELECT rowid, latitude, longitude, wind_dir_10m, wind_speed_10m,
           wind_dir_100m, wind_dir_10mMin, wind_dir_100mMin,
           wind_dir_10mMax, wind_dir_100mMax, temperature_2m, temperature_2m_min,
           temperature_2m_max, relative_humidity_2m, relative_humidity_2m_min,
           relative_humidity_2m_max, dewpoint_2m, dewpoint_2m_min, dewpoint_2m_max,
           rain, rain_min, rain_max, surface_pressure, surface_pressure_min,
           surface_pressure_max, et0_fao_evapotranspiration, et0_fao_evapotranspiration_min,
           et0_fao_evapotranspiration_max, vapour_pressure_deficit, vapour_pressure_deficit_min,
           vapour_pressure_deficit_max, wind_speed_10m_min,
           wind_speed_10m_max, wind_speed_100m, wind_speed_100m_min,
           wind_speed_100m_max, wind_gusts_10m, wind_gusts_10m_min,
           wind_gusts_10m_max, jourferieroudimanche
    FROM weather_data 
    WHERE deltadaydeath = 0 
    AND wind_dir_10m IS NOT NULL
    """
    
    all_weather = pd.read_sql_query(query, conn)
    print(f"Points météo à traiter: {len(all_weather)}")
    
    # Pour la prédiction, on utilise une direction moyenne vers les émissaires
    # ou on peut faire plusieurs prédictions avec différents bearings
    
    # Approche simplifiée: prédiction avec bearing=wind_direction (émissaire dans le vent)
    all_weather['bearing_to_emissaire'] = all_weather['wind_dir_10m']
    #all_weather['wind_emissaire_alignment'] = 0  # Parfait alignement
    
    # Ajouter les features dérivées
    if 'wind_dir_10mMin' in all_weather.columns and 'wind_dir_10mMax' in all_weather.columns:
        all_weather['wind_dir_amplitude'] = np.where(
            (all_weather['wind_dir_10mMax'] - all_weather['wind_dir_10mMin']) > 180,
            360 - (all_weather['wind_dir_10mMax'] - all_weather['wind_dir_10mMin']),
            all_weather['wind_dir_10mMax'] - all_weather['wind_dir_10mMin']
        )
    
    if 'wind_speed_10m_min' in all_weather.columns and 'wind_speed_10m_max' in all_weather.columns:
        all_weather['wind_speed_amplitude'] = all_weather['wind_speed_10m_max'] - all_weather['wind_speed_10m_min']
    
    if 'temperature_2m_min' in all_weather.columns and 'temperature_2m_max' in all_weather.columns:
        all_weather['temperature_amplitude'] = all_weather['temperature_2m_max'] - all_weather['temperature_2m_min']
    
    # Sélectionner les features dans l'ordre
    X_all = all_weather[feature_names]
    
    # Gérer les valeurs manquantes
    mask = ~X_all.isnull().any(axis=1)
    valid_data = all_weather[mask]
    X_valid = X_all[mask]
    
    print(f"Points valides pour prédiction: {len(X_valid)}")
    
    # Prédiction par chunks
    predictions = []
    chunk_size = 10000
    
    for i in range(0, len(X_valid), chunk_size):
        chunk = X_valid.iloc[i:i+chunk_size]
        pred_chunk = model.predict(chunk)
        predictions.extend(pred_chunk)
        
        if (i // chunk_size + 1) % 5 == 0:
            print(f"  Traité: {i + len(chunk)}/{len(X_valid)} points")
    
    # Mise à jour de la base
    print("Mise à jour de retro10m50km...")
    
    updates = [(pred, emissaire_altitude, int(rowid)) for pred, rowid in zip(predictions, valid_data['rowid'])]

    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE weather_data SET retro10m50km = ?, retro100m50km = ? WHERE rowid = ?",
        updates
    )
    conn.commit()
    conn.close()
    
    print(f"✓ {len(updates)} lignes mises à jour")
    
    return np.array(predictions)

def plot_feature_importance(model, feature_names):
    """Affiche l'importance des features."""
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    top_features = importance_df.tail(20)
    
    plt.figure(figsize=(12, 10))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title("Top 20 - Importance des features pour prédiction distance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_predictions_analysis(y_test, y_pred, weights=None):
    """Analyse des prédictions avec pondération par certitude."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prédictions vs réelles
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_test, y_pred, c=weights if weights is not None else 'blue', 
                         alpha=0.6, s=10, cmap='viridis')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Distances réelles (km)')
    ax1.set_ylabel('Distances prédites (km)')
    ax1.set_title('Prédictions vs Réelles (couleur = certitude)')
    if weights is not None:
        plt.colorbar(scatter, ax=ax1, label='Certitude')
    
    # Distribution des erreurs
    ax2 = axes[0, 1]
    errors = y_test.values - y_pred
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Erreur (réelle - prédite) km')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f'Distribution des erreurs (MAE: {np.mean(np.abs(errors)):.1f} km)')
    ax2.axvline(x=0, color='red', linestyle='--')
    
    # Erreur vs certitude
    ax3 = axes[1, 0]
    if weights is not None:
        ax3.scatter(weights, np.abs(errors), alpha=0.6, s=10)
        ax3.set_xlabel('Certitude')
        ax3.set_ylabel('Erreur absolue (km)')
        ax3.set_title('Erreur vs Certitude')
    else:
        ax3.text(0.5, 0.5, 'Pas de données de certitude', ha='center', va='center', transform=ax3.transAxes)
    
    # Distribution des distances prédites
    ax4 = axes[1, 1]
    ax4.hist(y_pred, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax4.set_xlabel('Distance prédite (km)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title(f'Distribution des prédictions (moy: {np.mean(y_pred):.1f} km)')
    
    plt.tight_layout()
    plt.show()

def main():
    weather_db_path = "/run/media/gael/GaelUSB/weather_data.db"
    emissaire_db_path = "/home/gael/Documents/cpp/cre/ClairVent/PyClairVent/Emissaire.db"
    
    print("=" * 90)
    print("PRÉDICTION DE DISTANCES AVEC CALCUL DE CERTITUDES BASÉES SUR LE VENT")
    print("=" * 90)
    
    # 1. Chargement des données
    print("\n[1/5] Chargement des données...")
    weather_data, emissaires = load_data_and_calculate_certainties(weather_db_path, emissaire_db_path)
    
    # 2. Calcul des certitudes et création du dataset
    print("\n[2/5] Calcul des certitudes et création du dataset d'entraînement...")
    training_df = create_training_dataset_with_certainties(weather_data, emissaires, certainty_threshold=0.3)
    
    if training_df.empty:
        print("❌ Impossible de créer le dataset d'entraînement")
        return
    
    # 3. Entraînement du modèle
    print("\n[3/5] Entraînement du modèle XGBoost...")
    model, feature_names, X_test, y_test, y_pred, w_test = train_distance_predictor_with_certainty(training_df)
    
    # 4. Prédiction pour tous les points météo
    print("\n[4/5] Application du modèle à tous les points météo...")
    all_predictions = predict_for_all_weather_points(model, feature_names, weather_db_path, emissaire_db_path)
    
    # 5. Visualisations
    print("\n[5/5] Génération des graphiques d'analyse...")
    plot_feature_importance(model, feature_names)
    plot_predictions_analysis(y_test, y_pred, w_test)
    
    # Statistiques finales
    print("\n" + "=" * 90)
    print("✓ ANALYSE TERMINÉE")
    print("=" * 90)
    
    print(f"\nRésultats de l'entraînement:")
    print(f"  - Échantillons d'entraînement: {len(training_df)}")
    print(f"  - Certitude moyenne: {training_df['certainty'].mean():.3f}")
    print(f"  - Distance moyenne d'entraînement: {training_df['target_distance'].mean():.1f} km")
    
    print(f"\nPrédictions finales:")
    print(f"  - Points météo mis à jour: {len(all_predictions)}")
    print(f"  - Distance moyenne prédite: {np.mean(all_predictions):.1f} km")
    print(f"  - Écart-type: {np.std(all_predictions):.1f} km")
    print(f"  - Min/Max: {np.min(all_predictions):.1f} - {np.max(all_predictions):.1f} km")

if __name__ == "__main__":
    main()

