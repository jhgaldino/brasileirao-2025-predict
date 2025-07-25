import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

# Funções auxiliares (iguais ao main)
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
                home_stats = None
                away_stats = None
                for team_name, team_stats in match['estatisticas'].items():
                    if team_name == home_team:
                        home_stats = team_stats
                    elif team_name == away_team:
                        away_stats = team_stats
                if not home_stats or not away_stats:
                    continue
                def get_stat(stats, key, default=0):
                    if key not in stats:
                        return default
                    value = stats[key]
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.strip('%'))
                    return float(value) if value else default
                row = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': match['partida']['placar']['mandante'],
                    'away_goals': match['partida']['placar']['visitante'],
                    'home_possession': get_stat(home_stats, 'posse_de_bola', 50),
                    'away_possession': get_stat(away_stats, 'posse_de_bola', 50),
                    'home_shots': get_stat(home_stats, 'chutes'),
                    'away_shots': get_stat(away_stats, 'chutes'),
                    'home_shots_target': get_stat(home_stats, 'chutes_a_gol'),
                    'away_shots_target': get_stat(away_stats, 'chutes_a_gol'),
                    'home_corners': get_stat(home_stats, 'escanteios'),
                    'away_corners': get_stat(away_stats, 'escanteios'),
                    'home_passes': get_stat(home_stats, 'passes'),
                    'away_passes': get_stat(away_stats, 'passes'),
                    'home_pass_accuracy': get_stat(home_stats, 'precisao_de_passe', 75),
                    'away_pass_accuracy': get_stat(away_stats, 'precisao_de_passe', 75),
                    'home_fouls': get_stat(home_stats, 'faltas'),
                    'away_fouls': get_stat(away_stats, 'faltas'),
                    'home_yellow_cards': get_stat(home_stats, 'cartoes_amarelos'),
                    'away_yellow_cards': get_stat(away_stats, 'cartoes_amarelos'),
                    'home_offsides': get_stat(home_stats, 'impedimentos'),
                    'away_offsides': get_stat(away_stats, 'impedimentos'),
                    'home_red_cards': get_stat(home_stats, 'cartoes_vermelhos'),
                    'away_red_cards': get_stat(away_stats, 'cartoes_vermelhos'),
                    'home_crosses': get_stat(home_stats, 'cruzamentos'),
                    'away_crosses': get_stat(away_stats, 'cruzamentos'),
                    'home_cross_accuracy': get_stat(home_stats, 'precisao_cruzamento', 25),
                    'away_cross_accuracy': get_stat(away_stats, 'precisao_cruzamento', 25),
                    'match_result': 'H' if match['partida']['placar']['mandante'] > match['partida']['placar']['visitante'] else 'A' if match['partida']['placar']['mandante'] < match['partida']['placar']['visitante'] else 'D'
                }
                matches_data.append(row)
        return pd.DataFrame(matches_data)
    except FileNotFoundError:
        print("Erro: Arquivo 'dataset.json' não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return None

def load_classification_data():
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar classificacao.json: {str(e)}")
        return None

def get_team_form(team_name, data):
    for team in data['classificacao']:
        if team['clube'] == team_name:
            form = team['ultimas_5']
            wins = form.count('V')
            draws = form.count('E')
            return (wins * 3 + draws) / 15
    return 0

def get_team_stats_from_table(team_name, classification_data):
    if not classification_data or 'classificacao' not in classification_data:
        return {}
    for team in classification_data['classificacao']:
        if team['clube'] == team_name:
            return {
                'posicao': team['posicao'],
                'pts': team['pts'],
                'vit': team['vit'],
                'e': team['e'],
                'der': team['der'],
                'gm': team['gm'],
                'gc': team['gc'],
                'sg': team['sg'],
                'pj': team['pj']
            }
    return {}

def get_last5_form(team_name, df, current_index):
    matches = df.iloc[:current_index]
    home = matches[matches['home_team'] == team_name]
    away = matches[matches['away_team'] == team_name]
    last5 = pd.concat([home, away]).sort_index().tail(5)
    if last5.empty:
        return 0.0
    points = 0
    for _, row in last5.iterrows():
        if row['home_team'] == team_name:
            if row['match_result'] == 'H':
                points += 3
            elif row['match_result'] == 'D':
                points += 1
        else:
            if row['match_result'] == 'A':
                points += 3
            elif row['match_result'] == 'D':
                points += 1
    return points / 15.0

def get_h2h_history(home_team, away_team, df, current_index):
    matches = df.iloc[:current_index]
    h2h = matches[((matches['home_team'] == home_team) & (matches['away_team'] == away_team)) |
                  ((matches['home_team'] == away_team) & (matches['away_team'] == home_team))]
    if h2h.empty:
        return 0.0
    home_wins = ((h2h['home_team'] == home_team) & (h2h['match_result'] == 'H')).sum()
    away_wins = ((h2h['away_team'] == home_team) & (h2h['match_result'] == 'A')).sum()
    total = len(h2h)
    return (home_wins + away_wins) / total if total > 0 else 0.0

def prepare_features(df, classification_data):
    classification_columns = [
        'home_position', 'away_position',
        'home_points', 'away_points',
        'home_wins', 'away_wins',
        'home_draws', 'away_draws',
        'home_losses', 'away_losses',
        'home_goals_scored', 'away_goals_scored',
        'home_goals_against', 'away_goals_against',
        'home_goal_diff', 'away_goal_diff',
        'home_played', 'away_played'
    ]
    for col in classification_columns:
        df[col] = 0
    df['home_last5_form'] = 0.0
    df['away_last5_form'] = 0.0
    df['h2h_home_win_rate'] = 0.0
    for idx, row in df.iterrows():
        home_stats = get_team_stats_from_table(row['home_team'], classification_data)
        away_stats = get_team_stats_from_table(row['away_team'], classification_data)
        if home_stats and away_stats:
            df.at[idx, 'home_position'] = home_stats.get('posicao', 0)
            df.at[idx, 'away_position'] = away_stats.get('posicao', 0)
            df.at[idx, 'home_points'] = home_stats.get('pts', 0)
            df.at[idx, 'away_points'] = away_stats.get('pts', 0)
            df.at[idx, 'home_wins'] = home_stats.get('vit', 0)
            df.at[idx, 'away_wins'] = away_stats.get('vit', 0)
            df.at[idx, 'home_draws'] = home_stats.get('e', 0)
            df.at[idx, 'away_draws'] = away_stats.get('e', 0)
            df.at[idx, 'home_losses'] = home_stats.get('der', 0)
            df.at[idx, 'away_losses'] = away_stats.get('der', 0)
            df.at[idx, 'home_goals_scored'] = home_stats.get('gm', 0)
            df.at[idx, 'away_goals_scored'] = away_stats.get('gm', 0)
            df.at[idx, 'home_goals_against'] = home_stats.get('gc', 0)
            df.at[idx, 'away_goals_against'] = away_stats.get('gc', 0)
            df.at[idx, 'home_goal_diff'] = home_stats.get('sg', 0)
            df.at[idx, 'away_goal_diff'] = away_stats.get('sg', 0)
            df.at[idx, 'home_played'] = home_stats.get('pj', 0)
            df.at[idx, 'away_played'] = away_stats.get('pj', 0)
        else:
            for col in classification_columns:
                df.at[idx, col] = 0
        df.at[idx, 'home_last5_form'] = get_last5_form(row['home_team'], df, idx)
        df.at[idx, 'away_last5_form'] = get_last5_form(row['away_team'], df, idx)
        df.at[idx, 'h2h_home_win_rate'] = get_h2h_history(row['home_team'], row['away_team'], df, idx)
    df['home_form'] = df['home_team'].apply(lambda x: get_team_form(x, classification_data))
    df['away_form'] = df['away_team'].apply(lambda x: get_team_form(x, classification_data))
    df['points_diff'] = df['home_points'] - df['away_points']
    df['position_diff'] = df['away_position'] - df['home_position']
    df['form_diff'] = df['home_form'] - df['away_form']
    df['goals_diff'] = df['home_goals_scored'] - df['away_goals_scored']
    df['strength_diff'] = (df['home_points'] / df['home_played'].replace(0,1)) - (df['away_points'] / df['away_played'].replace(0,1))
    df['form_momentum'] = df['home_form'] - df['away_form']
    df['home_win_rate'] = df['home_wins'] / df['home_played'].replace(0,1)
    df['away_win_rate'] = df['away_wins'] / df['away_played'].replace(0,1)
    df['home_scoring_rate'] = df['home_goals_scored'] / df['home_played'].replace(0,1)
    df['away_scoring_rate'] = df['away_goals_scored'] / df['away_played'].replace(0,1)
    features = [
        'home_possession', 'away_possession',
        'home_shots', 'away_shots',
        'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners',
        'home_passes', 'away_passes',
        'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls',
        'home_yellow_cards', 'away_yellow_cards',
        'home_offsides', 'away_offsides',
        'home_red_cards', 'away_red_cards',
        'home_crosses', 'away_crosses',
        'home_cross_accuracy', 'away_cross_accuracy'
    ] + classification_columns + [
        'strength_diff', 'form_momentum',
        'home_win_rate', 'away_win_rate',
        'home_scoring_rate', 'away_scoring_rate',
        'home_last5_form', 'away_last5_form', 'h2h_home_win_rate'
    ]
    return df[features], df['match_result']

def prepare_features_next_round(next_round, classification_data):
    # Prepara features para os jogos da próxima rodada
    rows = []
    for match in next_round['partidas']:
        home_team = match['mandante']
        away_team = match['visitante']
        home_stats = get_team_stats_from_table(home_team, classification_data)
        away_stats = get_team_stats_from_table(away_team, classification_data)
        row = {
            'home_team': home_team,
            'away_team': away_team,
            'home_possession': 50,
            'away_possession': 50,
            'home_shots': 10,
            'away_shots': 10,
            'home_shots_target': 4,
            'away_shots_target': 4,
            'home_corners': 5,
            'away_corners': 5,
            'home_passes': 400,
            'away_passes': 400,
            'home_pass_accuracy': 80,
            'away_pass_accuracy': 80,
            'home_fouls': 15,
            'away_fouls': 15,
            'home_yellow_cards': 2,
            'away_yellow_cards': 2,
            'home_offsides': 2,
            'away_offsides': 2,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'home_crosses': 15,
            'away_crosses': 15,
            'home_cross_accuracy': 30,
            'away_cross_accuracy': 30,
            'home_position': home_stats.get('posicao', 0),
            'away_position': away_stats.get('posicao', 0),
            'home_points': home_stats.get('pts', 0),
            'away_points': away_stats.get('pts', 0),
            'home_wins': home_stats.get('vit', 0),
            'away_wins': away_stats.get('vit', 0),
            'home_draws': home_stats.get('e', 0),
            'away_draws': away_stats.get('e', 0),
            'home_losses': home_stats.get('der', 0),
            'away_losses': away_stats.get('der', 0),
            'home_goals_scored': home_stats.get('gm', 0),
            'away_goals_scored': away_stats.get('gm', 0),
            'home_goals_against': home_stats.get('gc', 0),
            'away_goals_against': away_stats.get('gc', 0),
            'home_goal_diff': home_stats.get('sg', 0),
            'away_goal_diff': away_stats.get('sg', 0),
            'home_played': home_stats.get('pj', 0),
            'away_played': away_stats.get('pj', 0),
            'strength_diff': (home_stats.get('pts', 0) / home_stats.get('pj', 1)) - (away_stats.get('pts', 0) / away_stats.get('pj', 1)),
            'form_momentum': get_team_form(home_team, classification_data) - get_team_form(away_team, classification_data),
            'home_win_rate': home_stats.get('vit', 0) / home_stats.get('pj', 1) if home_stats.get('pj', 0) > 0 else 0,
            'away_win_rate': away_stats.get('vit', 0) / away_stats.get('pj', 1) if away_stats.get('pj', 0) > 0 else 0,
            'home_scoring_rate': home_stats.get('gm', 0) / home_stats.get('pj', 1) if home_stats.get('pj', 0) > 0 else 0,
            'away_scoring_rate': away_stats.get('gm', 0) / away_stats.get('pj', 1) if away_stats.get('pj', 0) > 0 else 0,
            'home_last5_form': 0.0,
            'away_last5_form': 0.0,
            'h2h_home_win_rate': 0.0
        }
        rows.append(row)
    features = [
        'home_possession', 'away_possession',
        'home_shots', 'away_shots',
        'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners',
        'home_passes', 'away_passes',
        'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls',
        'home_yellow_cards', 'away_yellow_cards',
        'home_offsides', 'away_offsides',
        'home_red_cards', 'away_red_cards',
        'home_crosses', 'away_crosses',
        'home_cross_accuracy', 'away_cross_accuracy',
        'home_position', 'away_position',
        'home_points', 'away_points',
        'home_wins', 'away_wins',
        'home_draws', 'away_draws',
        'home_losses', 'away_losses',
        'home_goals_scored', 'away_goals_scored',
        'home_goals_against', 'away_goals_against',
        'home_goal_diff', 'away_goal_diff',
        'home_played', 'away_played',
        'strength_diff', 'form_momentum',
        'home_win_rate', 'away_win_rate',
        'home_scoring_rate', 'away_scoring_rate',
        'home_last5_form', 'away_last5_form', 'h2h_home_win_rate'
    ]
    return pd.DataFrame(rows)[features]

def main():
    df = load_data()
    classification_data = load_classification_data()
    if df is not None and classification_data is not None:
        X, y = prepare_features(df, classification_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
        # Avaliação rápida
        print('Acurácia treino:', rf.score(X_train_resampled, y_train_resampled))
        print('Acurácia teste:', rf.score(scaler.transform(X_test), y_test))
        # Predição da próxima rodada
        try:
            with open('next_round.json', 'r', encoding='utf-8') as f:
                next_round = json.load(f)
            X_next = prepare_features_next_round(next_round, classification_data)
            X_next_scaled = scaler.transform(X_next)
            probs = rf.predict_proba(X_next_scaled)
            preds = rf.predict(X_next_scaled)
            print('\nPrevisão da próxima rodada:')
            for i, match in enumerate(next_round['partidas']):
                home = match['mandante']
                away = match['visitante']
                prob_h = probs[i][list(rf.classes_).index('H')] if 'H' in rf.classes_ else 0
                prob_d = probs[i][list(rf.classes_).index('D')] if 'D' in rf.classes_ else 0
                prob_a = probs[i][list(rf.classes_).index('A')] if 'A' in rf.classes_ else 0
                print(f"{home} x {away}: {preds[i]} (H: {prob_h:.2f} | D: {prob_d:.2f} | A: {prob_a:.2f})")
        except Exception as e:
            print(f'Erro ao prever próxima rodada: {e}')

if __name__ == "__main__":
    main() 