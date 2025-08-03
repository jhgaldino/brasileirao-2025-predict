import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier

# class labels
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}
def load_data():
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
        print("Erro: Arquivo 'dataset.json' não encontrado.")
        return None

def load_classification_data():
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar classificacao.json: {str(e)}")
        return None

def get_team_stats_from_table(team_name, classification_data):
    if not classification_data or 'classificacao' not in classification_data: return {}
    for team in classification_data['classificacao']:
        if team['clube'] == team_name:
            return team
    return {}

def get_team_form(team_name, data):
    for team in data.get('classificacao', []):
        if team['clube'] == team_name:
            form = team['ultimas_5']
            wins = form.count('V')
            draws = form.count('E')
            return (wins * 3 + draws) / 15
    return 0

def get_last5_form(team_name, df, current_index):
    matches = df.iloc[:current_index]
    home = matches[matches['home_team'] == team_name]
    away = matches[matches['away_team'] == team_name]
    last5 = pd.concat([home, away]).sort_index().tail(5)
    if last5.empty: return 0.0
    points = 0
    for _, row in last5.iterrows():
        if row['home_team'] == team_name:
            points += 3 if row['match_result'] == 'H' else 1 if row['match_result'] == 'D' else 0
        else:
            points += 3 if row['match_result'] == 'A' else 1 if row['match_result'] == 'D' else 0
    return points / 15.0

def get_h2h_history(home_team, away_team, df, current_index):
    matches = df.iloc[:current_index]
    h2h = matches[((matches['home_team'] == home_team) & (matches['away_team'] == away_team)) |
                  ((matches['home_team'] == away_team) & (matches['away_team'] == home_team))]
    if h2h.empty: return 0.0
    home_team_wins = ((h2h['home_team'] == home_team) & (h2h['match_result'] == 'H')).sum() + \
                     ((h2h['away_team'] == home_team) & (h2h['match_result'] == 'A')).sum()
    total = len(h2h)
    return home_team_wins / total if total > 0 else 0.0

def prepare_features(df):
    """
    Prepara as features de forma iterativa para evitar data leakage.
    Não usa mais o classification.json para dados históricos.
    """
    df = df.copy()

    # Mapeia chaves de estatísticas para nomes de colunas descritivos
    name_map = {
        'pts': 'points', 'pj': 'played', 'vit': 'wins', 'e': 'draws',
        'der': 'losses', 'gm': 'goals_scored', 'gc': 'goals_against', 'sg': 'goal_diff'
    }
    classification_columns = []
    for key in name_map.values():
        classification_columns.append(f'home_{key}')
        classification_columns.append(f'away_{key}')

    feature_columns = [
        'home_possession', 'away_possession', 'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners', 'home_passes', 'away_passes', 'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls', 'home_yellow_cards', 'away_yellow_cards', 'home_offsides', 'away_offsides',
        'home_red_cards', 'away_red_cards', 'home_crosses', 'away_crosses', 'home_cross_accuracy', 'away_cross_accuracy'
    ] + classification_columns + [
        'strength_diff', 'form_momentum', 'home_win_rate', 'away_win_rate',
        'home_scoring_rate', 'away_scoring_rate', 'home_last5_form', 'away_last5_form', 'h2h_home_win_rate',
        'home_form', 'away_form'
    ]

    # Inicializa colunas de features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Dicionário para manter o estado da classificação "ao vivo"
    live_stats = {}
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    for team in all_teams:
        live_stats[team] = {'pts': 0, 'pj': 0, 'vit': 0, 'e': 0, 'der': 0, 'gm': 0, 'gc': 0, 'sg': 0}

    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        # Preenche as features com os dados ANTES da partida atual
        for stat_key, col_suffix in name_map.items():
            df.loc[idx, f'home_{col_suffix}'] = live_stats[home_team][stat_key]
            df.loc[idx, f'away_{col_suffix}'] = live_stats[away_team][stat_key]

        # Calcula features derivadas com os dados pré-partida
        home_played = max(1, live_stats[home_team]['pj'])
        away_played = max(1, live_stats[away_team]['pj'])

        df.loc[idx, 'home_last5_form'] = get_last5_form(home_team, df, idx)
        df.loc[idx, 'away_last5_form'] = get_last5_form(away_team, df, idx)
        df.loc[idx, 'h2h_home_win_rate'] = get_h2h_history(home_team, away_team, df, idx)

        df.loc[idx, 'strength_diff'] = (live_stats[home_team]['pts'] / home_played) - (live_stats[away_team]['pts'] / away_played)
        df.loc[idx, 'form_momentum'] = df.loc[idx, 'home_last5_form'] - df.loc[idx, 'away_last5_form']
        df.loc[idx, 'home_win_rate'] = live_stats[home_team]['vit'] / home_played
        df.loc[idx, 'away_win_rate'] = live_stats[away_team]['vit'] / away_played
        df.loc[idx, 'home_scoring_rate'] = live_stats[home_team]['gm'] / home_played
        df.loc[idx, 'away_scoring_rate'] = live_stats[away_team]['gm'] / away_played

        # Atualiza as estatísticas "ao vivo" com o resultado da partida
        home_goals = int(row['home_shots_target'] * 0.3) # Simulação de gols para o cálculo
        away_goals = int(row['away_shots_target'] * 0.3)

        live_stats[home_team]['pj'] += 1
        live_stats[away_team]['pj'] += 1
        live_stats[home_team]['gm'] += home_goals
        live_stats[away_team]['gm'] += away_goals
        live_stats[home_team]['gc'] += away_goals
        live_stats[away_team]['gc'] += home_goals

        if row['match_result'] == 'H':
            live_stats[home_team]['pts'] += 3; live_stats[home_team]['vit'] += 1; live_stats[away_team]['der'] += 1
        elif row['match_result'] == 'A':
            live_stats[away_team]['pts'] += 3; live_stats[away_team]['vit'] += 1; live_stats[home_team]['der'] += 1
        else:
            live_stats[home_team]['pts'] += 1; live_stats[away_team]['pts'] += 1; live_stats[home_team]['e'] += 1; live_stats[away_team]['e'] += 1

    return df[feature_columns].fillna(0), df['match_result'], feature_columns


# ==============================================================================
# NOVAS FUNÇÕES ADICIONADAS PARA COMPLETAR O SCRIPT
# ==============================================================================

def train_gb_ros_model(X, y):
    """
    Treina o modelo Gradient Boosting com balanceamento RandomOverSampler.
    """
    print("Treinando o modelo Gradient Boosting com RandomOverSampler...")

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanceamento com RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
    
    # Mapeia os resultados (H, D, A) para números (0, 1, 2)
    y_resampled_num = pd.Series(y_resampled).map(CLASS_MAP).values

    # Treinamento do modelo
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, subsample=0.8)
    model.fit(X_resampled, y_resampled_num)
    
    print("Modelo treinado com sucesso.")
    return model, scaler

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

def predict_next_round(model, scaler, next_round_file, classification_data, historical_df, feature_columns):
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

    for match in next_round_data.get('partidas', []):
        home_team = match.get('mandante')
        away_team = match.get('visitante')

        if not home_team or not away_team:
            continue

        home_stats = get_team_stats_from_table(home_team, classification_data)
        away_stats = get_team_stats_from_table(away_team, classification_data)
        
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
            'home_position': home_stats.get('posicao', 20), 'away_position': away_stats.get('posicao', 20),
            'home_points': home_stats.get('pts', 0), 'away_points': away_stats.get('pts', 0),
            'home_wins': home_stats.get('vit', 0), 'away_wins': away_stats.get('vit', 0),
            'home_draws': home_stats.get('e', 0), 'away_draws': away_stats.get('e', 0),
            'home_losses': home_stats.get('der', 0), 'away_losses': away_stats.get('der', 0),
            'home_goals_scored': home_stats.get('gm', 0), 'away_goals_scored': away_stats.get('gm', 0),
            'home_goals_against': home_stats.get('gc', 0), 'away_goals_against': away_stats.get('gc', 0),
            'home_goal_diff': home_stats.get('sg', 0), 'away_goal_diff': away_stats.get('sg', 0),
            'home_played': home_stats.get('pj', 1), 'away_played': away_stats.get('pj', 1)
        })
        
        # Features calculadas
        home_played = max(1, row['home_played'])
        away_played = max(1, row['away_played'])
        row['home_form'] = get_team_form(home_team, classification_data)
        row['away_form'] = get_team_form(away_team, classification_data)
        row['strength_diff'] = (row['home_points'] / home_played) - (row['away_points'] / away_played)
        row['form_momentum'] = row['home_form'] - row['away_form']
        row['home_win_rate'] = row['home_wins'] / home_played
        row['away_win_rate'] = row['away_wins'] / away_played
        row['home_scoring_rate'] = row['home_goals_scored'] / home_played
        row['away_scoring_rate'] = row['away_goals_scored'] / away_played
        
        # Features de histórico (Last 5, H2H) - Usando uma aproximação para predição futura
        row['home_last5_form'] = row['home_form']
        row['away_last5_form'] = row['away_form'] 
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

def main():
    """
    Função principal que orquestra o processo de treinamento e predição.
    """
    # 1. Carregar dados históricos e de classificação
    df = load_data()
    classification_data = load_classification_data()

    if df is None or classification_data is None:
        print("It was not possible to load the data.")
        return
        
    # 2. Preparar features para treinamento
    X, y, feature_columns = prepare_features(df)
    
    # 3. Treinar o modelo GB + ROS
    model, scaler = train_gb_ros_model(X, y)
    
    # 4. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', classification_data, df, feature_columns)

if __name__ == "__main__":
    main()