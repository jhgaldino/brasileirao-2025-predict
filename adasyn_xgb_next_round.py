import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

# Mapeamento de classes para o XGBoost (que prefere alvos numéricos)
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

# ==============================================================================
# Função de Treinamento do Modelo ADASYN + XGBoost
# ==============================================================================

def train_adasyn_xgb_model(X, y):
    """
    Treina o modelo XGBoost com balanceamento ADASYN e hiperparâmetros otimizados.
    """
    print("Training the ADASYN + XGBoost model with optimized hyperparameters...")
    
    # Divisão em treino e teste para avaliação interna
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceamento com ADASYN
    print("Applying ADASYN for data balancing...")
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)

    # Mapeia os resultados (H, D, A) para números (0, 1, 2)
    y_train_res_num = pd.Series(y_train_resampled).map(CLASS_MAP).values
    y_test_num = pd.Series(y_test).map(CLASS_MAP).values
    
    # Definição dos melhores hiperparâmetros encontrados
    best_params = {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    # Treinamento do XGBoost com os hiperparâmetros otimizados
    print("Training XGBoost with optimized hyperparameters...")
    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(X_train_resampled, y_train_res_num)
    print("\nOptimized hyperparameters applied.")

    # Avaliação final no conjunto de teste para referência
    y_pred = best_xgb.predict(X_test_scaled)
    print("\nFinal Model Performance on Internal Test Set:")
    print(classification_report(y_test_num, y_pred, target_names=list(CLASS_MAP.keys())))
    
    return best_xgb, scaler


# ==============================================================================
# Função de Predição para a Próxima Rodada
# ==============================================================================

def predict_next_round(model, scaler, next_round_file, live_stats, historical_df, feature_columns):
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
    
    for match in next_round_data.get('partidas', []):
        home_team = match.get('mandante')
        away_team = match.get('visitante')

        if not home_team or not away_team:
            continue

        home_s = live_stats[home_team]
        away_s = live_stats[away_team]

        # Criação da linha de features para a nova partida
        row = {}        
        # Features de estatísticas (usando médias históricas do time)
        home_avg_stats = get_team_average_stats(home_team, historical_df)
        away_avg_stats = get_team_average_stats(away_team, historical_df)

        for stat, value in home_avg_stats.items():
            row[f'home_{stat}'] = value if not pd.isna(value) else 0
            row[f'away_{stat}'] = away_avg_stats.get(stat, 0) if not pd.isna(away_avg_stats.get(stat, 0)) else 0

        # Features da tabela de classificação
        row.update({
            'home_points': home_s.get('pts', 0), 'away_points': away_s.get('pts', 0),
            'home_wins': home_s.get('vit', 0), 'away_wins': away_s.get('vit', 0),
            'home_draws': home_s.get('e', 0), 'away_draws': away_s.get('e', 0),
            'home_losses': home_s.get('der', 0), 'away_losses': away_s.get('der', 0),
            'home_goals_scored': home_s.get('gm', 0), 'away_goals_scored': away_s.get('gm', 0),
            'home_goals_against': home_s.get('gc', 0), 'away_goals_against': away_s.get('gc', 0),
            'home_goal_diff': home_s.get('sg', 0), 'away_goal_diff': away_s.get('sg', 0),
            'home_played': home_s.get('pj', 1), 'away_played': away_s.get('pj', 1)
        })
        
        # Features calculadas
        home_played = max(1, row['home_played'])
        away_played = max(1, row['away_played'])
        row['home_last5_form'] = get_last5_form(home_team, historical_df, len(historical_df))
        row['away_last5_form'] = get_last5_form(away_team, historical_df, len(historical_df))
        row['strength_diff'] = (row['home_points'] / home_played) - (row['away_points'] / away_played)
        row['form_momentum'] = row['home_last5_form'] - row['away_last5_form']
        row['home_win_rate'] = row['home_wins'] / home_played
        row['away_win_rate'] = row['away_wins'] / away_played
        row['home_scoring_rate'] = row['home_goals_scored'] / home_played
        row['away_scoring_rate'] = row['away_goals_scored'] / away_played
        row['h2h_home_win_rate'] = get_h2h_history(home_team, away_team, historical_df, len(historical_df))

        # Criar DataFrame com as features na ordem correta
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

def get_team_average_stats(team_name, historical_df):
    """Calcula as estatísticas médias de um time com base nos dados históricos."""
    team_stats = {}
    # Coletando jogos em casa e fora
    home_games = historical_df[historical_df['home_team'] == team_name]
    away_games = historical_df[historical_df['away_team'] == team_name]

    for stat in ['possession', 'shots', 'shots_target', 'corners', 'passes', 'pass_accuracy', 'fouls', 'yellow_cards', 'offsides', 'red_cards', 'crosses', 'cross_accuracy']:
        home_mean = home_games[f'home_{stat}'].mean() if not home_games.empty else 0
        away_mean = away_games[f'away_{stat}'].mean() if not away_games.empty else 0
        team_stats[stat] = (home_mean + away_mean) / 2 # Média simples entre jogos em casa e fora
    return team_stats

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
    X, y, feature_columns, live_stats = prepare_features_iterative(df)
    
    # 3. Treinar o modelo ADASYN + XGBoost
    model, scaler = train_adasyn_xgb_model(X, y)
    
    # 4. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', live_stats, df, feature_columns)

if __name__ == "__main__":
    main()