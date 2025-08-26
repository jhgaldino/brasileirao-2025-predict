import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def load_data():
    try:
        with open('dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches_data = []
        # Iterate through rounds and matches
        for rodada in data['rodadas']:
            for match in rodada['partidas']:
                if 'estatisticas' not in match:
                    continue
                    
                home_team = match['partida']['mandante']
                away_team = match['partida']['visitante']
                
                # Get statistics for each team using team names
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
        print("Dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_team_form(team_name, data):
    """Calcula o aproveitamento recente do time baseado em ultimas_5"""
    for team in data['classificacao']:
        if team['clube'] == team_name:
            form = team['ultimas_5']
            wins = form.count('V')
            draws = form.count('E')
            return (wins * 3 + draws) / 15  # Máximo de 15 pontos possíveis
    return 0

def prepare_features(df, classification_data):
    # Initialize classification columns with zeros
    classification_columns = [
        'home_position', 'away_position',
        'home_points', 'away_points',
        'home_wins', 'away_wins',
        'home_draws', 'away_draws',
        'home_losses', 'away_losses',
        'home_goals_scored', 'away_goals_scored',
        'home_goals_against', 'away_goals_against',
        'home_goal_diff', 'away_goal_diff',
        'home_played', 'away_played'  # Adicionando jogos disputados
    ]
    
    for col in classification_columns:
        df[col] = 0

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
    ] + classification_columns
    
    # Add classification data
    for idx, row in df.iterrows():
        home_stats = get_team_stats_from_table(row['home_team'], classification_data)
        away_stats = get_team_stats_from_table(row['away_team'], classification_data)
        
        if home_stats and away_stats:
            df.at[idx, 'home_position'] = home_stats.get('posicao', 0) # Usar .get() com valor padrão
            df.at[idx, 'away_position'] = away_stats.get('posicao', 0)
            df.at[idx, 'home_points'] = home_stats.get('pts', 0) # Corrigido para 'pts'
            df.at[idx, 'away_points'] = away_stats.get('pts', 0)
            df.at[idx, 'home_wins'] = home_stats.get('vit', 0) # Corrigido para 'vit'
            df.at[idx, 'away_wins'] = away_stats.get('vit', 0)
            df.at[idx, 'home_draws'] = home_stats.get('e', 0) # Corrigido para 'e'
            df.at[idx, 'away_draws'] = away_stats.get('e', 0)
            df.at[idx, 'home_losses'] = home_stats.get('der', 0) # Corrigido para 'der'
            df.at[idx, 'away_losses'] = away_stats.get('der', 0)
            df.at[idx, 'home_goals_scored'] = home_stats.get('gm', 0) # Corrigido para 'gm'
            df.at[idx, 'away_goals_scored'] = away_stats.get('gm', 0)
            df.at[idx, 'home_goals_against'] = home_stats.get('gc', 0) # Corrigido para 'gc'
            df.at[idx, 'away_goals_against'] = away_stats.get('gc', 0)
            df.at[idx, 'home_goal_diff'] = home_stats.get('sg', 0) # Corrigido para 'sg'
            df.at[idx, 'away_goal_diff'] = away_stats.get('sg', 0)
            df.at[idx, 'home_played'] = home_stats.get('pj', 0)
            df.at[idx, 'away_played'] = away_stats.get('pj', 0)
        else:
            # Se não encontrar dados de classificação, preencher com 0 ou outro valor padrão
            for col in ['home_position', 'away_position', 'home_points', 'away_points',
                        'home_wins', 'away_wins', 'home_draws', 'away_draws',
                        'home_losses', 'away_losses', 'home_goals_scored', 'away_goals_scored',
                        'home_goals_against', 'away_goals_against', 'home_goal_diff', 'away_goal_diff',
                        'home_played', 'away_played']:
                df.at[idx, col] = 0 # Ou defina um valor padrão razoável

    # Adiciona features de momento
    df['home_form'] = df['home_team'].apply(lambda x: get_team_form(x, classification_data))
    df['away_form'] = df['away_team'].apply(lambda x: get_team_form(x, classification_data))
    
    # Adiciona features de diferença
    df['points_diff'] = df['home_points'] - df['away_points']
    df['position_diff'] = df['away_position'] - df['home_position']
    df['form_diff'] = df['home_form'] - df['away_form']
    df['goals_diff'] = df['home_goals_scored'] - df['away_goals_scored']
    
    # Add all the derived features
    df['strength_diff'] = (df['home_points'] / df['home_played']) - (df['away_points'] / df['away_played'])
    df['form_momentum'] = df['home_form'] - df['away_form']
    df['home_win_rate'] = df['home_wins'] / df['home_played']
    df['away_win_rate'] = df['away_wins'] / df['away_played']
    df['home_scoring_rate'] = df['home_goals_scored'] / df['home_played']
    df['away_scoring_rate'] = df['away_goals_scored'] / df['away_played']
    
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
        'home_scoring_rate', 'away_scoring_rate'
    ]
    
    X = df[features]
    y = df['match_result']
    
    return X, y

def train_model(X, y):
    # Define ALL features including the derived ones
    feature_columns = [
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
        # Additional derived features
        'strength_diff', 'form_momentum',
        'home_win_rate', 'away_win_rate',
        'home_scoring_rate', 'away_scoring_rate'
    ]
    
    # Ensure X is a copy to avoid SettingWithCopyWarning
    X = X.copy()
    
    # Calculate additional features for training data
    X.loc[:, 'strength_diff'] = (X['home_points'] / X['home_played']) - (X['away_points'] / X['away_played'])
    X.loc[:, 'home_win_rate'] = X['home_wins'] / X['home_played'] 
    X.loc[:, 'away_win_rate'] = X['away_wins'] / X['away_played']
    X.loc[:, 'home_scoring_rate'] = X['home_goals_scored'] / X['home_played']
    X.loc[:, 'away_scoring_rate'] = X['away_goals_scored'] / X['away_played']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance the classes using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # X_train_resampled and X_test_scaled are already numpy arrays from scaler and SMOTE
    # No need to convert back to DataFrame for fitting if we want to avoid feature name warnings
    # X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_columns)
    # X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    # X_train_resampled = X_train_resampled[feature_columns]
    # X_test_scaled = X_test_scaled[feature_columns]
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=400,  # Increased n_estimators
        max_depth=10,       # Increased max_depth
        min_samples_split=5, # Reduced min_samples_split
        min_samples_leaf=2,  # Reduced min_samples_leaf
        class_weight={
            'H': 1.0,
            'D': 1.2,
            'A': 1.0
        },
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, scaler, X_test_scaled, y_test

def train_ensemble(X, y):
    # Split, scale, and balance as in train_model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    
    rf.fit(X_train_resampled, y_train_resampled)
    xgb.fit(X_train_resampled, y_train_resampled)
    
    return lambda x: np.mean([
        rf.predict_proba(x),
        xgb.predict_proba(x)
    ], axis=0), scaler, X_test_scaled, y_test
    
    return lambda x: np.mean([
        rf.predict_proba(x),
        xgb.predict_proba(x)
    ], axis=0)

def evaluate_model(model, X, y):
    scores = cross_val_score(model, X.values, y, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {scores}")
    print(f"Average accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

def predict_next_round(model, scaler, next_round_file, classification_data, historical_team_data):
    try:
        # Load next round data
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round = json.load(f)

        predictions = []

        # Convert historical_team_data to DataFrame for easier lookup
        # Usaremos um DataFrame de Pandas para lookup eficiente
        # Assegure-se de que os nomes dos times são o índice do DataFrame
        historical_df = pd.DataFrame.from_dict(historical_team_data, orient='index')

        if not hasattr(model, 'classes_') or len(model.classes_) != 3:
            raise ValueError("Model not properly trained with all outcome classes (H/D/A)")

        all_match_features = [] # Lista para coletar features de todas as partidas

        # Garantir que todas as partidas sejam processadas
        for match in next_round['partidas']:
            home_team = match['mandante']
            away_team = match['visitante']

            # Get team classification stats
            home_stats_class = get_team_stats_from_table(home_team, classification_data) # Renomeado para evitar conflito com historical_stats
            away_stats_class = get_team_stats_from_table(away_team, classification_data) # Renomeado para evitar conflito com historical_stats

            if not home_stats_class or not away_stats_class:
                print(f"Warning: Missing classification statistics for {home_team} vs {away_team}. Skipping prediction for this match.")
                predictions.append(f"{home_team} vs {away_team}: No Prediction (Missing Classification Data)")
                continue
            
            if not home_stats_class or not home_stats_class.get('pts') or not away_stats_class or not away_stats_class.get('pts'):
                print(f"Warning: Missing classification statistics for {home_team} vs {away_team}. Skipping prediction for this match.")
                predictions.append(f"{home_team} vs {away_team}: No Prediction (Missing Classification Data)")
                continue

            # Obter estatísticas históricas específicas para cada time
            home_hist_stats = historical_df.loc[home_team] if home_team in historical_df.index else {}
            away_hist_stats = historical_df.loc[away_team] if away_team in historical_df.index else {}

            # Definir valores padrão para as estatísticas históricas caso não existam dados
            def get_hist_stat(team_hist_stats, stat_key, role, default_value):
                # Tenta pegar a média específica para a role (mandante/visitante)
                key = f"{stat_key}_media_as_{role}"
                if key in team_hist_stats and not pd.isna(team_hist_stats[key]):
                    return team_hist_stats[key]
                return default_value # Retorna um valor padrão se não encontrar ou for NaN

            # Calcule as features adicionais
            home_played = home_stats_class['pj']
            away_played = away_stats_class['pj']

            strength_diff = ((home_stats_class['pts'] / home_played) if home_played > 0 else 0) - \
                            ((away_stats_class['pts'] / away_played) if away_played > 0 else 0)
            form_momentum = get_team_form(home_team, classification_data) - get_team_form(away_team, classification_data)
            home_win_rate = (home_stats_class['vit'] / home_played) if home_played > 0 else 0
            away_win_rate = (away_stats_class['vit'] / away_played) if away_played > 0 else 0
            home_scoring_rate = (home_stats_class['gm'] / home_played) if home_played > 0 else 0
            away_scoring_rate = (away_stats_class['gm'] / away_played) if away_played > 0 else 0

            # Construir o dicionário de features para a partida atual
            row = {
                'home_possession': get_hist_stat(home_hist_stats, 'posse_de_bola', 'mandante', 50),
                'away_possession': get_hist_stat(away_hist_stats, 'posse_de_bola', 'visitante', 50),
                'home_shots': get_hist_stat(home_hist_stats, 'chutes', 'mandante', 10),
                'away_shots': get_hist_stat(away_hist_stats, 'chutes', 'visitante', 10),
                'home_shots_target': get_hist_stat(home_hist_stats, 'chutes_a_gol', 'mandante', 4),
                'away_shots_target': get_hist_stat(away_hist_stats, 'chutes_a_gol', 'visitante', 4),
                'home_corners': get_hist_stat(home_hist_stats, 'escanteios', 'mandante', 5),
                'away_corners': get_hist_stat(away_hist_stats, 'escanteios', 'visitante', 5),
                'home_passes': get_hist_stat(home_hist_stats, 'passes', 'mandante', 400),
                'away_passes': get_hist_stat(away_hist_stats, 'passes', 'visitante', 400),
                'home_pass_accuracy': get_hist_stat(home_hist_stats, 'precisao_de_passe', 'mandante', 80),
                'away_pass_accuracy': get_hist_stat(away_hist_stats, 'precisao_de_passe', 'visitante', 80),
                'home_fouls': get_hist_stat(home_hist_stats, 'faltas', 'mandante', 15),
                'away_fouls': get_hist_stat(away_hist_stats, 'faltas', 'visitante', 15),
                'home_yellow_cards': get_hist_stat(home_hist_stats, 'cartoes_amarelos', 'mandante', 2),
                'away_yellow_cards': get_hist_stat(away_hist_stats, 'cartoes_amarelos', 'visitante', 2),
                'home_offsides': get_hist_stat(home_hist_stats, 'impedimentos', 'mandante', 2),
                'away_offsides': get_hist_stat(away_hist_stats, 'impedimentos', 'visitante', 2),
                'home_red_cards': get_hist_stat(home_hist_stats, 'cartoes_vermelhos', 'mandante', 0),
                'away_red_cards': get_hist_stat(away_hist_stats, 'cartoes_vermelhos', 'visitante', 0),
                'home_crosses': get_hist_stat(home_hist_stats, 'cruzamentos', 'mandante', 15),
                'away_crosses': get_hist_stat(away_hist_stats, 'cruzamentos', 'visitante', 15),
                'home_cross_accuracy': get_hist_stat(home_hist_stats, 'precisao_cruzamento', 'mandante', 30),
                'away_cross_accuracy': get_hist_stat(away_hist_stats, 'precisao_cruzamento', 'visitante', 30),

                # Classification stats (mantido, pois já usa home_stats_class e away_stats_class)
                'home_position': home_stats_class['posicao'],
                'away_position': away_stats_class['posicao'],
                'home_points': home_stats_class['pts'],
                'away_points': away_stats_class['pts'],
                'home_wins': home_stats_class['vit'],
                'away_wins': away_stats_class['vit'],
                'home_draws': home_stats_class['e'],
                'away_draws': away_stats_class['e'],
                'home_losses': home_stats_class['der'],
                'away_losses': away_stats_class['der'],
                'home_goals_scored': home_stats_class['gm'],
                'away_goals_scored': away_stats_class['gm'],
                'home_goals_against': home_stats_class['gc'],
                'away_goals_against': away_stats_class['gc'],
                'home_goal_diff': home_stats_class['sg'],
                'away_goal_diff': away_stats_class['sg'],
                'home_played': home_stats_class['pj'],
                'away_played': away_stats_class['pj'],
                'strength_diff': strength_diff,
                'form_momentum': form_momentum,
                'home_win_rate': home_win_rate,
                'away_win_rate': away_win_rate,
                'home_scoring_rate': home_scoring_rate,
                'away_scoring_rate': away_scoring_rate
            }
            all_match_features.append(row) # Adiciona o dicionário de features à lista

        # Criar o DataFrame de features a partir da lista de dicionários
        features = pd.DataFrame(all_match_features)

        # If no matches were processed, return an empty list
        if features.empty:
            print("Warning: No matches could be processed due to missing classification data.")
            return ["No predictions available: Missing classification data for all matches."]

        # Assegurar que a ordem das colunas é a mesma do treinamento (IMPORTANTE!)
        # Use feature_names_in_ para obter a ordem das features do model treinado
        if hasattr(model, 'feature_names_in_'):
            X_train_cols = model.feature_names_in_
            # Reindexa as colunas para garantir a ordem correta
            features = features.reindex(columns=X_train_cols, fill_value=0)
        else:
            print("Warning: Could not determine feature order from the model. This might cause issues if feature order differs.")

        scaled_features = scaler.transform(features)
        probabilities = model.predict_proba(scaled_features)

        # Ajuste para obter os resultados corretamente do classes_ do model
        class_to_index = {cls: idx for idx, cls in enumerate(model.classes_)}
        home_index = class_to_index.get('H', -1)
        draw_index = class_to_index.get('D', -1)
        away_index = class_to_index.get('A', -1)

        for i, match_data in enumerate(next_round['partidas']):
            home_team = match_data['mandante']
            away_team = match_data['visitante']

            prob_home_win = probabilities[i][home_index] if home_index != -1 else 0
            prob_draw = probabilities[i][draw_index] if draw_index != -1 else 0
            prob_away_win = probabilities[i][away_index] if away_index != -1 else 0

            # Se todas as probabilidades são 0, pular a previsão
            if prob_home_win == 0 and prob_draw == 0 and prob_away_win == 0:
                predictions.append(f"{home_team} vs {away_team}: No Prediction (Probabilities are all zero)")
                continue

            # Mais um pequeno ajuste para pegar a classe predita diretamente
            predicted_class_label = model.classes_[np.argmax(probabilities[i])]

            # More balanced thresholds
            home_threshold = 0.38
            away_threshold = 0.42
            draw_threshold = 0.30

            # Lógica de decisão
            if predicted_class_label == 'D' and prob_draw >= draw_threshold:
                final_prediction = 'Draw'
            elif predicted_class_label == 'H' and prob_home_win >= home_threshold:
                final_prediction = 'Home Win'
            elif predicted_class_label == 'A' and prob_away_win >= away_threshold:
                final_prediction = 'Away Win'
            else:
                # Fallback para a classe com maior probabilidade se os thresholds não forem atingidos
                final_prediction = {
                    'H': 'Home Win',
                    'D': 'Draw',
                    'A': 'Away Win'
                }.get(predicted_class_label, 'Unknown')


            predictions.append(f"{home_team} vs {away_team}: {final_prediction} (H:{prob_home_win:.2f}/D:{prob_draw:.2f}/A:{prob_away_win:.2f})")

        return predictions

    except FileNotFoundError:
        print(f"Error: {next_round_file} not found")
        return None
    except Exception as e:
        print(f"Error processing predictions: {str(e)}")
        return None

def get_team_stats_from_table(team_name, classification_data): # Adicionado classification_data como parâmetro
    """Obtém estatísticas de classificação de um time a partir dos dados carregados."""
    # Remover o bloco try-except com a abertura do arquivo, pois os dados já estão carregados
    # e passados como argumento.
    
    if not classification_data or 'classificacao' not in classification_data:
        # Se os dados de classificação não foram carregados ou estão incompletos, retorna dicionário vazio.
        return {}
    
    for team in classification_data['classificacao']: # Usa o classification_data passado
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
    # Este print pode ser útil para depuração, mas o warning de "Missing classification data" no predict_next_round já indica o problema.
    # print(f"Warning: Team '{team_name}' not found in classification data.")
    return {} # Retorna dicionário vazio se o time não for encontrado no classification_data

def get_team_historical_stats(team_name, dataset):
    """Obtém médias específicas do time"""
    team_stats = []
    for rodada in dataset['rodadas']:
        for match in rodada['partidas']:
            if team_name in match['estatisticas']:
                team_stats.append(match['estatisticas'][team_name])
    
    if not team_stats:
        return None
        
    return {
        'avg_possession': np.mean([float(s['posse_de_bola'].strip('%')) for s in team_stats]),
        'avg_shots': np.mean([s['chutes'] for s in team_stats]),
        'avg_shots_target': np.mean([s['chutes_a_gol'] for s in team_stats]),
        # ... outras estatísticas
    }

def load_classification_data():
    """Load team classification data from JSON file"""
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading classification data: {str(e)}")
        return None

def get_team_strength_score(team_name, data):
    """Calculate overall team strength based on multiple factors"""
    team_data = next(team for team in data['classificacao'] if team['clube'] == team_name)
    
    points_per_game = team_data['pts'] / team_data['pj']
    goal_diff_per_game = team_data['sg'] / team_data['pj']
    recent_form = sum(4 if x == 'V' else 1 if x == 'E' else 0 for x in team_data['ultimas_5']) / 5
    
    return (points_per_game * 0.5) + (goal_diff_per_game * 0.3) + (recent_form * 0.2)

def calculate_team_averages():
    df = load_data()
    if df is None:
        return {}
    
    team_stats = {}
    for _, row in df.iterrows():
        # Home team stats
        if row['home_team'] not in team_stats:
            team_stats[row['home_team']] = {'home_games': 0, 'away_games': 0}  # Initialize both
        team_stats[row['home_team']]['home_games'] += 1
        
        # Away team stats
        if row['away_team'] not in team_stats:
            team_stats[row['away_team']] = {'home_games': 0, 'away_games': 0}  # Initialize both
        team_stats[row['away_team']]['away_games'] += 1
        
        # Calculate averages for each stat
        for stat in ['possession', 'shots', 'shots_target', 'corners', 'passes', 
                     'pass_accuracy', 'fouls', 'yellow_cards', 'offsides', 
                     'red_cards', 'crosses', 'cross_accuracy']:
            # Home stats
            home_key = f'home_{stat}'
            if home_key in row:
                if f'{stat}_total_home' not in team_stats[row['home_team']]:
                    team_stats[row['home_team']][f'{stat}_total_home'] = 0
                team_stats[row['home_team']][f'{stat}_total_home'] += row[home_key]
            
            # Away stats
            away_key = f'away_{stat}'
            if away_key in row:
                if f'{stat}_total_away' not in team_stats[row['away_team']]:
                    team_stats[row['away_team']][f'{stat}_total_away'] = 0
                team_stats[row['away_team']][f'{stat}_total_away'] += row[away_key]
    
    # Calculate averages
    for team in team_stats:
        for stat in ['possession', 'shots', 'shots_target', 'corners', 'passes', 
                     'pass_accuracy', 'fouls', 'yellow_cards', 'offsides', 
                     'red_cards', 'crosses', 'cross_accuracy']:
            if f'{stat}_total_home' in team_stats[team]:
                team_stats[team][f'{stat}_media_as_mandante'] = (
                    team_stats[team][f'{stat}_total_home'] / team_stats[team]['home_games']
                )
            if f'{stat}_total_away' in team_stats[team]:
                team_stats[team][f'{stat}_media_as_visitante'] = (
                    team_stats[team][f'{stat}_total_away'] / team_stats[team]['away_games']
                )
    
    return team_stats

def main():
    # Load and prepare data
    df = load_data()
    classification_data = load_classification_data()

    # Calculate historical team averages
    historical_team_data = calculate_team_averages()

    if df is not None and classification_data is not None:
        X, y = prepare_features(df, classification_data)

        # Train model
        model, scaler, _, _ = train_model(X, y)

        # Evaluate model
        evaluate_model(model, X, y)

        # Print feature importances
        importances = model.feature_importances_
        feature_names = X.columns

        print("\nFeature importances:")
        for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance}")

        # Make predictions for next round
        predictions = predict_next_round(model, scaler, 'next_round.json', classification_data, historical_team_data, feature_names)
        if predictions:
            for pred in predictions:
                print(pred)

if __name__ == "__main__":
    main()