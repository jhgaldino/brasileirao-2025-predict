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
        for match in data['partidas']:
            home_team = match['partida']['mandante']
            away_team = match['partida']['visitante']
            home_stats = match['estatisticas'][home_team]
            away_stats = match['estatisticas'][away_team]
            
            def get_stat(stats, key, default=0):
                if key in stats:
                    value = stats[key]
                    if isinstance(value, str) and value.endswith('%'):
                        return int(value.strip('%'))
                    return value
                return default
            
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
    ]
    
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
        with open(next_round_file, 'r', encoding='utf-8') as f:
            next_round = json.load(f)
        
        predictions = []
        historical_stats = {
            'home_possession': 55,
            'away_possession': 45,
            'home_shots': 12,
            'away_shots': 8,
            'home_shots_target': 5,
            'away_shots_target': 3,
            'home_corners': 6,
            'away_corners': 4,
            'home_passes': 450,
            'away_passes': 350,
            'home_pass_accuracy': 80,
            'away_pass_accuracy': 75,
            'home_fouls': 15,
            'away_fouls': 15,
            'home_yellow_cards': 2,
            'away_yellow_cards': 2,
            'home_offsides': 2,
            'away_offsides': 2,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'home_crosses': 15,
            'away_crosses': 10,
            'home_cross_accuracy': 30,
            'away_cross_accuracy': 25
        }
        
        # Verify model classes
        if not hasattr(model, 'classes_') or len(model.classes_) != 3:
            raise ValueError("Model not properly trained with all outcome classes (H/D/A)")
            
        for match in next_round['partidas']:
            features = np.array([[
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
                historical_stats['away_cross_accuracy']
            ]])
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            result_map = {
                'H': 'Home Win',
                'D': 'Draw',
                'A': 'Away Win'
            }
            
            predictions.append({
                'home_team': match['mandante'],
                'away_team': match['visitante'],
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

def main():
    # Load and prepare data
    df = load_data()
    if df is None:
        return
    
    X, y = prepare_features(df)
    model, scaler, X_test_scaled, y_test = train_model(X, y)
    
    # Model evaluation
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.2f}\n")
    
    # Predict next round
    predictions = predict_next_round(model, scaler, 'next_round.json')
    
    # Print predictions
    print("Predictions for next round:")
    print("=" * 50)
    for pred in predictions:
        print(f"\n{pred['home_team']} vs {pred['away_team']}")
        print(f"Predicted result: {pred['predicted_result']}")
        print(f"Win probability: {pred['home_win_prob']}")
        print(f"Draw probability: {pred['draw_prob']}")
        print(f"Loss probability: {pred['away_win_prob']}")

if __name__ == "__main__":
    main()