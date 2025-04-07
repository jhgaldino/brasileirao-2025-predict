import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

def prepare_features(df):
    # Initialize classification columns with zeros
    classification_columns = [
        'home_position', 'away_position',
        'home_points', 'away_points',
        'home_wins', 'away_wins',
        'home_draws', 'away_draws',
        'home_losses', 'away_losses',
        'home_goals_scored', 'away_goals_scored',
        'home_goals_against', 'away_goals_against',
        'home_goal_diff', 'away_goal_diff'
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
        home_stats = get_team_stats_from_table(row['home_team'])
        away_stats = get_team_stats_from_table(row['away_team'])
        
        if home_stats and away_stats:
            df.at[idx, 'home_position'] = home_stats['posicao']
            df.at[idx, 'away_position'] = away_stats['posicao']
            df.at[idx, 'home_points'] = home_stats['pontos']
            df.at[idx, 'away_points'] = away_stats['pontos']
            df.at[idx, 'home_wins'] = home_stats['vitorias']
            df.at[idx, 'away_wins'] = away_stats['vitorias']
            df.at[idx, 'home_draws'] = home_stats['empates']
            df.at[idx, 'away_draws'] = away_stats['empates']
            df.at[idx, 'home_losses'] = home_stats['derrotas']
            df.at[idx, 'away_losses'] = away_stats['derrotas']
            df.at[idx, 'home_goals_scored'] = home_stats['gols_marcados']
            df.at[idx, 'away_goals_scored'] = away_stats['gols_marcados']
            df.at[idx, 'home_goals_against'] = home_stats['gols_sofridos']
            df.at[idx, 'away_goals_against'] = away_stats['gols_sofridos']
            df.at[idx, 'home_goal_diff'] = home_stats['saldo_gols']
            df.at[idx, 'away_goal_diff'] = away_stats['saldo_gols']
    
    X = df[features]
    y = df['match_result']
    
    return X, y

def train_model(X, y):
    # Convert pandas DataFrame/Series to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # Ensure we have all possible classes in the training data
    unique_classes = np.unique(y)
    expected_classes = ['H', 'D', 'A']
    
    if not all(cls in unique_classes for cls in expected_classes):
        print("Warning: Not all match outcomes (H/D/A) present in training data")
        # Add dummy samples if needed
        missing_classes = set(expected_classes) - set(unique_classes)
        for cls in missing_classes:
            X = np.vstack([X, X.mean(axis=0).reshape(1, -1)])
            y = np.append(y, cls)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

def predict_next_round(model, scaler, next_round_file):
    try:
        # Load next round data
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round = json.load(f)
            
        # Load historical stats
        from calcular_media_historica import calcular_medias
        historical_stats = calcular_medias()
        
        predictions = []
        
        if not hasattr(model, 'classes_') or len(model.classes_) != 3:
            raise ValueError("Model not properly trained with all outcome classes (H/D/A)")
            
        # Garantir que todas as partidas sejam processadas
        for match in next_round['partidas']:
            home_team = match['mandante']
            away_team = match['visitante']
            
            # Get team classification stats
            home_stats = get_team_stats_from_table(home_team)
            away_stats = get_team_stats_from_table(away_team)
            
            if not home_stats or not away_stats:
                print(f"Warning: Missing statistics for {home_team} vs {away_team}")
                continue
                
            features = np.array([
                [
                    historical_stats['home_possession'],
                    historical_stats['away_possession'],
                    historical_stats['home_shots'],
                    historical_stats['away_shots'],
                    historical_stats['home_shots_target'],
                    historical_stats['away_shots_target'],
                    historical_stats['home_corners'],
                    historical_stats['away_corners'],
                    historical_stats['home_passes'],
                    historical_stats['away_passes'],
                    historical_stats['home_pass_accuracy'],
                    historical_stats['away_pass_accuracy'],
                    historical_stats['home_fouls'],
                    historical_stats['away_fouls'],
                    historical_stats['home_yellow_cards'],
                    historical_stats['away_yellow_cards'],
                    historical_stats['home_offsides'],
                    historical_stats['away_offsides'],
                    historical_stats['home_red_cards'],
                    historical_stats['away_red_cards'],
                    historical_stats['home_crosses'],
                    historical_stats['away_crosses'],
                    historical_stats['home_cross_accuracy'],
                    historical_stats['away_cross_accuracy'],
                    home_stats['posicao'],
                    away_stats['posicao'],
                    home_stats['pontos'],
                    away_stats['pontos'],
                    home_stats['vitorias'],
                    away_stats['vitorias'],
                    home_stats['empates'],
                    away_stats['empates'],
                    home_stats['derrotas'],
                    away_stats['derrotas'],
                    home_stats['gols_marcados'],
                    away_stats['gols_marcados'],
                    home_stats['gols_sofridos'],
                    away_stats['gols_sofridos'],
                    home_stats['saldo_gols'],
                    away_stats['saldo_gols']
                ]
            ])
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            result_map = {
                'H': 'Home Win',
                'D': 'Draw',
                'A': 'Away Win'
            }
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'predicted_result': result_map[prediction],
                'home_win_prob': f"{prob[0]:.2f}",
                'draw_prob': f"{prob[1]:.2f}",
                'away_win_prob': f"{prob[2]:.2f}"
            })
        
        return predictions
        
    except FileNotFoundError:
        print(f"Error: {next_round_file} not found")
        return None
    except Exception as e:
        print(f"Error processing predictions: {str(e)}")
        return None

def get_team_stats_from_table(team_name):
    try:
        with open('classificacao.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for team in data['classificacao']:
            if team['time'] == team_name:
                return {
                    'posicao': team['posicao'],
                    'pontos': team['pontos'],
                    'vitorias': team['vitorias'],
                    'empates': team['empates'],
                    'derrotas': team['derrotas'],
                    'gols_marcados': team['gols_marcados'],
                    'gols_sofridos': team['gols_sofridos'],
                    'saldo_gols': team['saldo_gols']
                }
        return None
    except Exception as e:
        print(f"Error loading classification data: {str(e)}")
        return None

def main():
    # Load and prepare data
    df = load_data()
    if df is not None:
        X, y = prepare_features(df)
        
        # Train model
        model, scaler, X_test, y_test = train_model(X, y)
        
        # Make predictions for next round
        predictions = predict_next_round(model, scaler, 'next_round.json')
        if predictions:
            for pred in predictions:
                print(f"{pred['home_team']} vs {pred['away_team']}: {pred['predicted_result']}")

if __name__ == "__main__":
    main()