import pandas as pd
import json

CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

def load_dataset(filepath='dataset.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
                    'home_goals': match['partida']['placar']['mandante'],
                    'away_goals': match['partida']['placar']['visitante'],
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
        print(f"Erro: Arquivo '{filepath}' não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar dados de {filepath}: {str(e)}")
        return None

def load_classification_data(filepath='classificacao.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar {filepath}: {str(e)}")
        return None

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
    home_team_wins = (((h2h['home_team'] == home_team) & (h2h['match_result'] == 'H')).sum() +
                     ((h2h['away_team'] == home_team) & (h2h['match_result'] == 'A')).sum())
    total = len(h2h)
    return home_team_wins / total if total > 0 else 0.0

def prepare_features_iterative(df, ewma_span=5):
    df = df.copy()

    # Dicionários para manter o estado "ao vivo"
    live_stats = {}
    live_ewma_stats = {}
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()

    # Nomes das estatísticas para cálculo da EWMA
    stat_features = [
        'possession', 'shots', 'shots_target', 'corners', 'passes', 'pass_accuracy',
        'fouls', 'yellow_cards', 'offsides', 'red_cards', 'crosses', 'cross_accuracy'
    ]

    # Inicializa os dicionários de estado
    for team in all_teams:
        live_stats[team] = {'pts': 0, 'pj': 0, 'vit': 0, 'e': 0, 'der': 0, 'gm': 0, 'gc': 0, 'sg': 0}
        live_ewma_stats[team] = {stat: 0.0 for stat in stat_features}

    # Inicializa as colunas de features no DataFrame
    for stat in stat_features:
        df[f'home_{stat}_ewma'] = 0.0
        df[f'away_{stat}_ewma'] = 0.0

    # Itera sobre o dataframe para calcular as features de forma temporalmente correta
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        # 1. Atribui as features com os dados ANTES da partida atual
        for stat in stat_features:
            df.loc[idx, f'home_{stat}_ewma'] = live_ewma_stats[home_team][stat]
            df.loc[idx, f'away_{stat}_ewma'] = live_ewma_stats[away_team][stat]

        # ... (cálculo de outras features como antes)
        home_played = max(1, live_stats[home_team]['pj'])
        away_played = max(1, live_stats[away_team]['pj'])
        df.loc[idx, 'home_last5_form'] = get_last5_form(home_team, df, idx)
        df.loc[idx, 'away_last5_form'] = get_last5_form(away_team, df, idx)
        df.loc[idx, 'h2h_home_win_rate'] = get_h2h_history(home_team, away_team, df, idx)
        df.loc[idx, 'strength_diff'] = (live_stats[home_team]['pts'] / home_played) - (live_stats[away_team]['pts'] / away_played)
        df.loc[idx, 'form_momentum'] = df.loc[idx, 'home_last5_form'] - df.loc[idx, 'away_last5_form']

        # 2. Atualiza os dicionários de estado com os resultados da partida atual
        # Atualiza EWMA
        alpha = 2 / (ewma_span + 1)
        for stat in stat_features:
            # Atualiza time da casa
            current_home_stat = row[f'home_{stat}']
            old_home_ewma = live_ewma_stats[home_team][stat]
            if old_home_ewma == 0.0: # Primeira partida do time
                live_ewma_stats[home_team][stat] = current_home_stat
            else:
                live_ewma_stats[home_team][stat] = alpha * current_home_stat + (1 - alpha) * old_home_ewma
            
            # Atualiza time visitante
            current_away_stat = row[f'away_{stat}']
            old_away_ewma = live_ewma_stats[away_team][stat]
            if old_away_ewma == 0.0: # Primeira partida do time
                live_ewma_stats[away_team][stat] = current_away_stat
            else:
                live_ewma_stats[away_team][stat] = alpha * current_away_stat + (1 - alpha) * old_away_ewma

        # Atualiza estatísticas da tabela
        home_goals = row['home_goals']
        away_goals = row['away_goals']
        live_stats[home_team]['pj'] += 1; live_stats[away_team]['pj'] += 1
        live_stats[home_team]['gm'] += home_goals; live_stats[away_team]['gm'] += away_goals
        live_stats[home_team]['gc'] += away_goals; live_stats[away_team]['gc'] += home_goals
        live_stats[home_team]['sg'] = live_stats[home_team]['gm'] - live_stats[home_team]['gc']
        live_stats[away_team]['sg'] = live_stats[away_team]['gm'] - live_stats[away_team]['gc']
        if row['match_result'] == 'H':
            live_stats[home_team]['pts'] += 3; live_stats[home_team]['vit'] += 1; live_stats[away_team]['der'] += 1
        elif row['match_result'] == 'A':
            live_stats[away_team]['pts'] += 3; live_stats[away_team]['vit'] += 1; live_stats[home_team]['der'] += 1
        else:
            live_stats[home_team]['pts'] += 1; live_stats[away_team]['pts'] += 1; live_stats[home_team]['e'] += 1; live_stats[away_team]['e'] += 1

    # Define as colunas de features a serem usadas no modelo
    feature_columns = ([f'home_{stat}_ewma' for stat in stat_features] +
                      [f'away_{stat}_ewma' for stat in stat_features] +
                      ['strength_diff', 'form_momentum', 'home_last5_form', 'away_last5_form', 'h2h_home_win_rate'])

    return df[feature_columns].fillna(0), df['match_result'], feature_columns, live_stats, live_ewma_stats