import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN  # Apenas ADASYN é importado
from xgboost import XGBClassifier         # Apenas XGBoost é importado
from sklearn.metrics import classification_report

# Mapeamento de classes para o XGBoost (que prefere alvos numéricos)
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

# ==============================================================================
# Funções de Carregamento e Preparação de Dados
# ==============================================================================

def load_data():
    """Carrega os dados históricos das partidas do dataset.json."""
    try:
        with open('dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        matches_data = []
        for rodada in data['rodadas']:
            for match in rodada['partidas']:
                if 'estatisticas' not in match:
                    continue
                home_team = match['partida']['mandante']
                away_team = match['partida']['visitante']
                home_stats = match['estatisticas'].get(home_team)
                away_stats = match['estatisticas'].get(away_team)
                if not home_stats or not away_stats:
                    continue
                
                def get_stat(stats, key, default=0):
                    if key not in stats or not stats[key]: return default
                    value = stats[key]
                    return float(str(value).strip('%')) if isinstance(value, str) else float(value)

                row = {
                    'home_team': home_team, 'away_team': away_team,
                    'home_possession': get_stat(home_stats, 'posse_de_bola', 50),
                    'away_possession': get_stat(away_stats, 'posse_de_bola', 50),
                    'home_shots': get_stat(home_stats, 'chutes'), 'away_shots': get_stat(away_stats, 'chutes'),
                    'home_shots_target': get_stat(home_stats, 'chutes_a_gol'), 'away_shots_target': get_stat(away_stats, 'chutes_a_gol'),
                    'home_corners': get_stat(home_stats, 'escanteios'), 'away_corners': get_stat(away_stats, 'escanteios'),
                    'home_passes': get_stat(home_stats, 'passes'), 'away_passes': get_stat(away_stats, 'passes'),
                    'home_pass_accuracy': get_stat(home_stats, 'precisao_de_passe', 75), 'away_pass_accuracy': get_stat(away_stats, 'precisao_de_passe', 75),
                    'home_fouls': get_stat(home_stats, 'faltas'), 'away_fouls': get_stat(away_stats, 'faltas'),
                    'home_yellow_cards': get_stat(home_stats, 'cartoes_amarelos'), 'away_yellow_cards': get_stat(away_stats, 'cartoes_amarelos'),
                    'home_offsides': get_stat(home_stats, 'impedimentos'), 'away_offsides': get_stat(away_stats, 'impedimentos'),
                    'home_red_cards': get_stat(home_stats, 'cartoes_vermelhos'), 'away_red_cards': get_stat(away_stats, 'cartoes_vermelhos'),
                    'home_crosses': get_stat(home_stats, 'cruzamentos'), 'away_crosses': get_stat(away_stats, 'cruzamentos'),
                    'home_cross_accuracy': get_stat(home_stats, 'precisao_cruzamento', 25), 'away_cross_accuracy': get_stat(away_stats, 'precisao_cruzamento', 25),
                    'match_result': 'H' if match['partida']['placar']['mandante'] > match['partida']['placar']['visitante'] else 'A' if match['partida']['placar']['mandante'] < match['partida']['placar']['visitante'] else 'D'
                }
                matches_data.append(row)
        return pd.DataFrame(matches_data)
    except FileNotFoundError:
        print("Error: 'dataset.json' file not found.")
        return None

def load_classification_data():
    """Carrega os dados da tabela de classificação do classificacao.json."""
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading classificacao.json: {str(e)}")
        return None

def get_team_stats_from_table(team_name, classification_data):
    """Busca as estatísticas de um time na tabela de classificação."""
    if not classification_data or 'classificacao' not in classification_data: return {}
    for team in classification_data['classificacao']:
        if team['clube'] == team_name:
            return team
    return {}

def prepare_features(df, classification_data):
    """Prepara o dataframe com todas as features para o treinamento."""
    # Adiciona colunas de classificação
    for idx, row in df.iterrows():
        home_stats = get_team_stats_from_table(row['home_team'], classification_data)
        away_stats = get_team_stats_from_table(row['away_team'], classification_data)
        
        # Adiciona stats do time da casa
        df.loc[idx, 'home_position'] = home_stats.get('posicao', 20)
        df.loc[idx, 'home_points'] = home_stats.get('pts', 0)
        df.loc[idx, 'home_played'] = home_stats.get('pj', 1)
        
        # Adiciona stats do time visitante
        df.loc[idx, 'away_position'] = away_stats.get('posicao', 20)
        df.loc[idx, 'away_points'] = away_stats.get('pts', 0)
        df.loc[idx, 'away_played'] = away_stats.get('pj', 1)

    # Cria features de diferença
    df['points_diff'] = df['home_points'] - df['away_points']
    df['position_diff'] = df['away_position'] - df['home_position'] # Posição menor é melhor

    # Cria features de média de pontos (força)
    home_played = df['home_played'].replace(0, 1)
    away_played = df['away_played'].replace(0, 1)
    df['home_strength'] = df['home_points'] / home_played
    df['away_strength'] = df['away_points'] / away_played
    df['strength_diff'] = df['home_strength'] - df['away_strength']
    
    feature_columns = [
        'home_possession', 'away_possession', 'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners', 'home_passes', 'away_passes', 'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls', 'home_yellow_cards', 'away_yellow_cards', 'home_offsides', 'away_offsides',
        'home_red_cards', 'away_red_cards', 'home_crosses', 'away_crosses', 'home_cross_accuracy', 'away_cross_accuracy',
        'home_position', 'away_position', 'home_points', 'away_points', 'home_played', 'away_played',
        'points_diff', 'position_diff', 'home_strength', 'away_strength', 'strength_diff'
    ]
    
    X = df[feature_columns].fillna(0)
    y = df['match_result']
    
    return X, y

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

def predict_next_round(model, scaler, next_round_file, classification_data):
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

        home_stats = get_team_stats_from_table(home_team, classification_data)
        away_stats = get_team_stats_from_table(away_team, classification_data)
        
        row = {
            'home_possession': 52, 'away_possession': 48, 'home_shots': 12, 'away_shots': 10,
            'home_shots_target': 4, 'away_shots_target': 3, 'home_corners': 6, 'away_corners': 4,
            'home_passes': 400, 'away_passes': 350, 'home_pass_accuracy': 80, 'away_pass_accuracy': 78,
            'home_fouls': 15, 'away_fouls': 16, 'home_yellow_cards': 2.5, 'away_yellow_cards': 2.7,
            'home_offsides': 2, 'away_offsides': 2.1, 'home_red_cards': 0.1, 'away_red_cards': 0.15,
            'home_crosses': 18, 'away_crosses': 15, 'home_cross_accuracy': 25, 'away_cross_accuracy': 23,
            'home_position': home_stats.get('posicao', 20), 'away_position': away_stats.get('posicao', 20),
            'home_points': home_stats.get('pts', 0), 'away_points': away_stats.get('pts', 0),
            'home_played': home_stats.get('pj', 1), 'away_played': away_stats.get('pj', 1)
        }
        
        row['points_diff'] = row['home_points'] - row['away_points']
        row['position_diff'] = row['away_position'] - row['home_position']
        row['home_strength'] = row['home_points'] / max(1, row['home_played'])
        row['away_strength'] = row['away_points'] / max(1, row['away_played'])
        row['strength_diff'] = row['home_strength'] - row['away_strength']

        features_df = pd.DataFrame([row])
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
    # 1. Carregar dados históricos e de classificação
    df = load_data()
    classification_data = load_classification_data()

    if df is None or classification_data is None:
        print("Could not load data. Exiting.")
        return
        
    # 2. Preparar features para treinamento
    X, y = prepare_features(df, classification_data)
    
    # 3. Treinar o modelo ADASYN + XGBoost
    model, scaler = train_adasyn_xgb_model(X, y)
    
    # 4. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', classification_data)

if __name__ == "__main__":
    main()