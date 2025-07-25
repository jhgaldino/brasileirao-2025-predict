import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek 
from xgboost import XGBClassifier


def load_data():
    try:
        with open('dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches_data = []
        # Itera através das rodadas e partidas
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
                    'match_result': 'H' if match['partida']['placar']['mandante'] > match['partida']['placar']['visitante']
                                else 'A' if match['partida']['placar']['mandante'] < match['partida']['placar']['visitante']
                                else 'D'
                }
                matches_data.append(row)
        
        return pd.DataFrame(matches_data)
    except FileNotFoundError:
        print("Arquivo dataset.json não encontrado. Por favor, verifique o caminho do arquivo.")
        return None
    except Exception as e:
        print(f"Erro ao carregar os dados: {str(e)}")
        return None

def load_classification_data():
    """Carrega os dados de classificação dos times do arquivo JSON."""
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar dados de classificação: {str(e)}")
        return None

def get_team_stats_from_table(team_name, classification_data):
    if not classification_data or 'classificacao' not in classification_data:
        return {}
    
    for team in classification_data['classificacao']:
        if team['clube'] == team_name:
            return {
                'posicao': team['posicao'], 'pts': team['pts'], 'vit': team['vit'],
                'e': team['e'], 'der': team['der'], 'gm': team['gm'],
                'gc': team['gc'], 'sg': team['sg'], 'pj': team['pj']
            }
    return {}

def get_team_form(team_name, data):
    for team in data['classificacao']:
        if team['clube'] == team_name:
            form = team['ultimas_5']
            wins = form.count('V')
            draws = form.count('E')
            return (wins * 3 + draws) / 15
    return 0

def prepare_features(df, classification_data):
    classification_columns = [
        'home_position', 'away_position', 'home_points', 'away_points',
        'home_wins', 'away_wins', 'home_draws', 'away_draws', 'home_losses', 'away_losses',
        'home_goals_scored', 'away_goals_scored', 'home_goals_against', 'away_goals_against',
        'home_goal_diff', 'away_goal_diff', 'home_played', 'away_played'
    ]
    for col in classification_columns:
        df[col] = 0

    # Adiciona dados da classificação ao DataFrame
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

    # Adiciona features de momento e força
    df['home_form'] = df['home_team'].apply(lambda x: get_team_form(x, classification_data))
    df['away_form'] = df['away_team'].apply(lambda x: get_team_form(x, classification_data))
    
    # Adiciona features de diferença para o model capturar o contraste entre as equipes
    df['points_diff'] = df['home_points'] - df['away_points']
    df['position_diff'] = df['away_position'] - df['home_position']
    
    # Tratamento para evitar divisão por zero
    df['home_played'].replace(0, 1, inplace=True)
    df['away_played'].replace(0, 1, inplace=True)

    df['strength_diff'] = (df['home_points'] / df['home_played']) - (df['away_points'] / df['away_played'])
    df['home_win_rate'] = df['home_wins'] / df['home_played']
    df['away_win_rate'] = df['away_wins'] / df['away_played']
    df['home_scoring_rate'] = df['home_goals_scored'] / df['home_played']
    df['away_scoring_rate'] = df['away_goals_scored'] / df['away_played']
    
    features = [
        'home_possession', 'away_possession', 'home_shots', 'away_shots',
        'home_shots_target', 'away_shots_target', 'home_corners', 'away_corners',
        'home_passes', 'away_passes', 'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls', 'home_yellow_cards', 'away_yellow_cards',
        'home_offsides', 'away_offsides', 'home_red_cards', 'away_red_cards',
        'home_crosses', 'away_crosses', 'home_cross_accuracy', 'away_cross_accuracy'
    ] + classification_columns + [
        'strength_diff', 'home_win_rate', 'away_win_rate', 
        'home_scoring_rate', 'away_scoring_rate', 'points_diff', 'position_diff'
    ]
    
    X = df[features].fillna(0) # Preenche possíveis NaNs com 0
    y = df['match_result']
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Normalização das features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Balanceamento de classes com SMOTETomek
    print("Aplicando SMOTETomek para balancear os dados de treino...")
    smotetomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_scaled, y_train)
    print("Balanceamento concluído.")
    
    # Converte de volta para DataFrame para manter os nomes das colunas
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 4. Treinamento do model RandomForest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, scaler, X_test_scaled, y_test

# ==============================================================================
# The following functions are used to evaluate the model and make predictions
# based on the trained model with SMOTETomek.
# ==============================================================================

def evaluate_model(model, X, y):
    """Avalia o model usando validação cruzada."""
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\nScores de Validação Cruzada: {scores}")
    print(f"Acurácia Média: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

def calculate_team_averages():
    """Calculate historical averages for each team based on their home and away statistics."""
    df = load_data()
    if df is None:
        return {}
    
    team_stats = {}
    stats_cols = ['possession', 'shots', 'shots_target', 'corners', 'passes', 
                  'pass_accuracy', 'fouls', 'yellow_cards', 'offsides', 
                  'red_cards', 'crosses', 'cross_accuracy']

    # Inicializa a estrutura de dados
    all_teams = set(df['home_team']).union(set(df['away_team']))
    for team in all_teams:
        team_stats[team] = {
            'home_games': 0, 'away_games': 0,
            **{f'{stat}_total_home': 0 for stat in stats_cols},
            **{f'{stat}_total_away': 0 for stat in stats_cols}
        }

    # accumulates statistics for each team
    for _, row in df.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        team_stats[home_team]['home_games'] += 1
        team_stats[away_team]['away_games'] += 1
        for stat in stats_cols:
            team_stats[home_team][f'{stat}_total_home'] += row.get(f'home_{stat}', 0)
            team_stats[away_team][f'{stat}_total_away'] += row.get(f'away_{stat}', 0)
    
    # Average calculations
    for team, data in team_stats.items():
        for stat in stats_cols:
            if data['home_games'] > 0:
                data[f'{stat}_media_as_mandante'] = data[f'{stat}_total_home'] / data['home_games']
            if data['away_games'] > 0:
                data[f'{stat}_media_as_visitante'] = data[f'{stat}_total_away'] / data['away_games']
    
    return team_stats

def predict_next_round(model, scaler, next_round_file, classification_data, historical_team_data):
    """
    Predicts the outcomes of the next round of matches using the trained model.
    This function reads the next round data from a JSON file, prepares the features
    using historical averages and classification data, and returns predictions for each match.
    """
    try:
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round = json.load(f)

        predictions = []
        all_match_features = []
        historical_df = pd.DataFrame.from_dict(historical_team_data, orient='index')

        for match in next_round['partidas']:
            home_team, away_team = match['mandante'], match['visitante']
            home_stats_class = get_team_stats_from_table(home_team, classification_data)
            away_stats_class = get_team_stats_from_table(away_team, classification_data)

            if not home_stats_class or not away_stats_class:
                print(f"Aviso: Faltam estatísticas de classificação para {home_team} vs {away_team}.")
                continue

            home_hist_stats = historical_df.loc[home_team] if home_team in historical_df.index else pd.Series()
            away_hist_stats = historical_df.loc[away_team] if away_team in historical_df.index else pd.Series()
            
            def get_hist_stat(team_hist_stats, stat_key, role, default):
                return team_hist_stats.get(f'{stat_key}_media_as_{role}', default)

            home_played = home_stats_class.get('pj', 1) or 1
            away_played = away_stats_class.get('pj', 1) or 1
            
            row = {
                'home_possession': get_hist_stat(home_hist_stats, 'possession', 'mandante', 50),
                'away_possession': get_hist_stat(away_hist_stats, 'possession', 'visitante', 50),
                # ... (outras estatísticas com valores padrão)
                'home_shots': get_hist_stat(home_hist_stats, 'shots', 'mandante', 10),
                'away_shots': get_hist_stat(away_hist_stats, 'shots', 'visitante', 8),
                'home_shots_target': get_hist_stat(home_hist_stats, 'shots_target', 'mandante', 4),
                'away_shots_target': get_hist_stat(away_hist_stats, 'shots_target', 'visitante', 3),
                'home_corners': get_hist_stat(home_hist_stats, 'corners', 'mandante', 5),
                'away_corners': get_hist_stat(away_hist_stats, 'corners', 'visitante', 4),
                'home_passes': get_hist_stat(home_hist_stats, 'passes', 'mandante', 400),
                'away_passes': get_hist_stat(away_hist_stats, 'passes', 'visitante', 350),
                'home_pass_accuracy': get_hist_stat(home_hist_stats, 'pass_accuracy', 'mandante', 80),
                'away_pass_accuracy': get_hist_stat(away_hist_stats, 'pass_accuracy', 'visitante', 78),
                'home_fouls': get_hist_stat(home_hist_stats, 'fouls', 'mandante', 15),
                'away_fouls': get_hist_stat(away_hist_stats, 'fouls', 'visitante', 16),
                'home_yellow_cards': get_hist_stat(home_hist_stats, 'yellow_cards', 'mandante', 2),
                'away_yellow_cards': get_hist_stat(away_hist_stats, 'yellow_cards', 'visitante', 2.5),
                'home_offsides': get_hist_stat(home_hist_stats, 'offsides', 'mandante', 2),
                'away_offsides': get_hist_stat(away_hist_stats, 'offsides', 'visitante', 2),
                'home_red_cards': get_hist_stat(home_hist_stats, 'red_cards', 'mandante', 0.1),
                'away_red_cards': get_hist_stat(away_hist_stats, 'red_cards', 'visitante', 0.1),
                'home_crosses': get_hist_stat(home_hist_stats, 'crosses', 'mandante', 15),
                'away_crosses': get_hist_stat(away_hist_stats, 'crosses', 'visitante', 12),
                'home_cross_accuracy': get_hist_stat(home_hist_stats, 'cross_accuracy', 'mandante', 30),
                'away_cross_accuracy': get_hist_stat(away_hist_stats, 'cross_accuracy', 'visitante', 28),

                # Estatísticas de classificação
                'home_position': home_stats_class['posicao'], 'away_position': away_stats_class['posicao'],
                'home_points': home_stats_class['pts'], 'away_points': away_stats_class['pts'],
                'home_wins': home_stats_class['vit'], 'away_wins': away_stats_class['vit'],
                'home_draws': home_stats_class['e'], 'away_draws': away_stats_class['e'],
                'home_losses': home_stats_class['der'], 'away_losses': away_stats_class['der'],
                'home_goals_scored': home_stats_class['gm'], 'away_goals_scored': away_stats_class['gm'],
                'home_goals_against': home_stats_class['gc'], 'away_goals_against': away_stats_class['gc'],
                'home_goal_diff': home_stats_class['sg'], 'away_goal_diff': away_stats_class['sg'],
                'home_played': home_played, 'away_played': away_played,

                # Features de diferença
                'points_diff': home_stats_class['pts'] - away_stats_class['pts'],
                'position_diff': away_stats_class['posicao'] - home_stats_class['posicao'],
                'strength_diff': (home_stats_class['pts'] / home_played) - (away_stats_class['pts'] / away_played),
                'home_win_rate': home_stats_class['vit'] / home_played,
                'away_win_rate': away_stats_class['vit'] / away_played,
                'home_scoring_rate': home_stats_class['gm'] / home_played,
                'away_scoring_rate': away_stats_class['gm'] / away_played,
            }
            all_match_features.append(row)

        if not all_match_features:
            print("Nenhuma partida pôde ser processada.")
            return []

        features_df = pd.DataFrame(all_match_features)
        
        # Garante a mesma ordem de colunas do treino
        features_df = features_df.reindex(columns=model.feature_names_in_, fill_value=0)

        scaled_features = scaler.transform(features_df)
        probabilities = model.predict_proba(scaled_features)

        class_map = {cls: i for i, cls in enumerate(model.classes_)}

        for i, match in enumerate(next_round['partidas']):
            home, away = match['mandante'], match['visitante']
            prob_H = probabilities[i][class_map.get('H', 0)]
            prob_D = probabilities[i][class_map.get('D', 1)]
            prob_A = probabilities[i][class_map.get('A', 2)]
            
            # Lógica de decisão simples baseada na maior probabilidade
            prediction = model.classes_[np.argmax(probabilities[i])]
            result_map = {'H': 'Vitória Mandante', 'D': 'Empate', 'A': 'Vitória Visitante'}
            
            predictions.append(
                f"{home} vs {away}: {result_map[prediction]} "
                f"(H: {prob_H:.2%}, D: {prob_D:.2%}, A: {prob_A:.2%})"
            )

        return predictions

    except FileNotFoundError:
        print(f"Erro: Arquivo {next_round_file} não encontrado")
        return None
    except Exception as e:
        print(f"Erro ao processar predições: {str(e)}")
        return None


def main():
    """Main function to execute the prediction workflow."""
    # Load data
    df = load_data()
    classification_data = load_classification_data()

    # Calculate historical averages for teams
    historical_team_data = calculate_team_averages()

    if df is not None and classification_data is not None:
        X, y = prepare_features(df, classification_data)
        
        # Train the model with SMOTETomek
        model, scaler, X_test, y_test = train_model(X, y)
        
        # Avaliate the model
        evaluate_model(model, X, y)
        
        # Show feature importances
        importances = model.feature_importances_
        feature_names = X.columns
        print("\nImportância das Features (model com SMOTETomek):")
        for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]:
            print(f"- {feature}: {importance:.4f}")
            
        # Do predictions for the next round
        print("\n--- Previsões para a Próxima Rodada ---")
        predictions = predict_next_round(model, scaler, 'next_round.json', classification_data, historical_team_data)
        if predictions:
            for pred in predictions:
                print(pred)

if __name__ == "__main__":
    main()