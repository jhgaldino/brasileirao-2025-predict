import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

# Mapeamento de classes
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

# ==============================================================================
# Função de Treinamento do Modelo SMOTE + MLP
# ==============================================================================

def train_smote_mlp_model(X, y):
    """
    Treina o modelo MLPClassifier com balanceamento SMOTE.
    """
    print("Training the SMOTE + MLP model...")
    
    # Divisão em treino e teste para avaliação interna
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceamento com SMOTE
    print("Applying SMOTE for data balancing...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Mapeia os resultados (H, D, A) para números (0, 1, 2)
    y_train_res_num = pd.Series(y_train_resampled).map(CLASS_MAP).values
    y_test_num = pd.Series(y_test).map(CLASS_MAP).values
    
    # Definição do modelo MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42, 
                        activation='relu', solver='adam', alpha=0.001295942459383017, learning_rate_init=0.08589138242660839)
    
    # Treinamento do MLP
    print("Training MLP Classifier...")
    mlp.fit(X_train_resampled, y_train_res_num)
    print("\nMLP model trained.")

    # Avaliação final no conjunto de teste para referência
    y_pred = mlp.predict(X_test_scaled)
    print("\nFinal Model Performance on Internal Test Set:")
    print(classification_report(y_test_num, y_pred, target_names=list(CLASS_MAP.keys())))
    
    return mlp, scaler

# ==============================================================================
# Função de Predição para a Próxima Rodada
# ==============================================================================

def predict_next_round(model, scaler, next_round_file, live_stats, live_ewma_stats, historical_df, feature_columns):
    """
    Prevê os resultados para as partidas no arquivo next_round.json.
    """
    print("\n--- Generating predictions for the next round ---")
    try:
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{next_round_file}' not found.")
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

# ==============================================================================
# Função Principal de Execução
# ==============================================================================

def main():
    """
    Orquestra o processo de treinamento e predição.
    """
    # 1. Carregar dados históricos
    df = load_dataset()

    if df is None:
        print("Could not load data. Exiting.")
        return
        
    # 2. Preparar features para treinamento
    X, y, feature_columns, live_stats, live_ewma_stats = prepare_features_iterative(df)
    
    # 3. Treinar o modelo SMOTE + MLP
    model, scaler = train_smote_mlp_model(X, y)
    
    # 4. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', live_stats, live_ewma_stats, df, feature_columns)

if __name__ == "__main__":
    main()