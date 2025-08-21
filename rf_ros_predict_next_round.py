import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

def train_rf_model(X, y):
    """
    Treina o modelo RandomForest com balanceamento RandomOverSampler.
    """
    print("Treinando o modelo RandomForest com RandomOverSampler...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=151, max_depth=10, random_state=42, min_samples_split=2, min_samples_leaf=2)
    rf.fit(X_train_resampled, y_train_resampled)
    # Avaliação rápida
    print('Acurácia treino:', rf.score(X_train_resampled, y_train_resampled))
    print('Acurácia teste:', rf.score(scaler.transform(X_test), y_test))
    return rf, scaler

def predict_next_round(model, scaler, next_round_file, live_stats, live_ewma_stats, historical_df, feature_columns):
    """
    Prevê os resultados para as partidas no arquivo next_round.json.
    """
    print("\n--- Generating predictions for the next round ---")
    try:
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{next_round_file}' not found.")
        return

    predictions = []
    stat_features = list(live_ewma_stats[list(live_ewma_stats.keys())[0]].keys())

    for match in next_round_data.get('partidas', []):
        home_team = match.get('mandante')
        away_team = match.get('visitante')

        if not home_team or not away_team:
            print(f"Skipping match with missing team names: {match}")
            continue

        if home_team not in live_stats or away_team not in live_stats:
            print(f"Skipping match because a team is not in live_stats: {home_team} or {away_team}")
            continue

        row = {}
        # Features de EWMA
        for stat in stat_features:
            row[f'home_{stat}_ewma'] = live_ewma_stats[home_team][stat]
            row[f'away_{stat}_ewma'] = live_ewma_stats[away_team][stat]

        # Features de classificação e outras
        home_s = live_stats[home_team]
        away_s = live_stats[away_team]
        home_played = max(1, home_s['pj'])
        away_played = max(1, away_s['pj'])

        row['home_last5_form'] = get_last5_form(home_team, historical_df, len(historical_df))
        row['away_last5_form'] = get_last5_form(away_team, historical_df, len(historical_df))
        row['strength_diff'] = (home_s['pts'] / home_played) - (away_s['pts'] / away_played)
        row['form_momentum'] = row['home_last5_form'] - row['away_last5_form']
        row['h2h_home_win_rate'] = get_h2h_history(home_team, away_team, historical_df, len(historical_df))

        features_df = pd.DataFrame([row])[feature_columns].fillna(0)
        features_scaled = scaler.transform(features_df)
        
        probs = model.predict_proba(features_scaled)
        preds = model.predict(features_scaled)[0]

        prob_h = probs[0][list(model.classes_).index('H')] if 'H' in model.classes_ else 0
        prob_d = probs[0][list(model.classes_).index('D')] if 'D' in model.classes_ else 0
        prob_a = probs[0][list(model.classes_).index('A')] if 'A' in model.classes_ else 0
        print(f"{home_team} x {away_team}: {preds} (H: {prob_h:.2f} | D: {prob_d:.2f} | A: {prob_a:.2f})")

def main():
    df = load_dataset()
    if df is not None:
        X, y, feature_columns, live_stats, live_ewma_stats = prepare_features_iterative(df)
        model, scaler = train_rf_model(X, y)
        predict_next_round(model, scaler, 'next_round.json', live_stats, live_ewma_stats, df, feature_columns)

if __name__ == "__main__":
    main()
