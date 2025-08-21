import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

# class labels
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

# ==============================================================================
# NOVAS FUNÇÕES ADICIONADAS PARA COMPLETAR O SCRIPT
# ==============================================================================

def train_gb_smote_model(X, y):
    """
    Treina o modelo Gradient Boosting com balanceamento SMOTE.
    """
    print("Treinando o modelo Gradient Boosting com SMOTE...")

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Mapeia os resultados (H, D, A) para números (0, 1, 2)
    y_resampled_num = pd.Series(y_resampled).map(CLASS_MAP).values

    # Treinamento do modelo
    model = GradientBoostingClassifier(n_estimators=364, learning_rate=0.12265764356910785, max_depth=4, random_state=42, subsample=0.7047898756660642)
    model.fit(X_resampled, y_resampled_num)
    
    print("Modelo treinado com sucesso.")
    return model, scaler

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
        
        probabilities = model.predict_proba(features_scaled)[0]
        
        prob_H = probabilities[CLASS_MAP['H']]
        prob_D = probabilities[CLASS_MAP['D']]
        prob_A = probabilities[CLASS_MAP['A']]
        
        prediction_text = f"{home_team} vs {away_team} -> H: {prob_H:.1%}, D: {prob_D:.1%}, A: {prob_A:.1%}"
        predictions.append(prediction_text)

    for pred in predictions:
        print(pred)

def main():
    """
    Função principal que orquestra o processo de treinamento e predição.
    """
    # 1. Carregar dados históricos
    df = load_dataset()

    if df is None:
        print("It was not possible to load the data.")
        return
        
    # 2. Preparar features para treinamento
    X, y, feature_columns, live_stats, live_ewma_stats = prepare_features_iterative(df)
    
    # 3. Treinar o modelo GB + SMOTE
    model, scaler = train_gb_smote_model(X, y)
    
    # 4. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', live_stats, live_ewma_stats, df, feature_columns)

if __name__ == "__main__":
    main()