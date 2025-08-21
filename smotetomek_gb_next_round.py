import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Aplicando SMOTETomek para balancear os dados de treino...")
    smotetomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_scaled, y_train)
    print("Balanceamento concluído.")
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Train the model
    model = GradientBoostingClassifier(
        n_estimators=235,
        learning_rate=0.20737738732010347,
        max_depth=4,
        random_state=42,
        subsample=0.7016566351370807
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

def get_team_average_stats(team_name, historical_df):
    team_stats = {} 
    home_games = historical_df[historical_df['home_team'] == team_name]
    away_games = historical_df[historical_df['away_team'] == team_name]

    for stat in ['possession', 'shots', 'shots_target', 'corners', 'passes', 'pass_accuracy', 'fouls', 'yellow_cards', 'offsides', 'red_cards', 'crosses', 'cross_accuracy']:
        home_mean = home_games[f'home_{stat}'].mean() if not home_games.empty else 0
        away_mean = away_games[f'away_{stat}'].mean() if not away_games.empty else 0
        team_stats[stat] = (home_mean + away_mean) / 2 # Média simples entre jogos em casa e fora
    return team_stats

def predict_next_round(model, scaler, next_round_file, live_stats, historical_df, feature_columns):
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

        for match in next_round['partidas']:
            home_team, away_team = match['mandante'], match['visitante']
            home_s = live_stats[home_team]
            away_s = live_stats[away_team]

            if not home_s or not away_s:
                print(f"Aviso: Faltam estatísticas de classificação para {home_team} vs {away_team}.")
                continue

            home_avg_stats = get_team_average_stats(home_team, historical_df)
            away_avg_stats = get_team_average_stats(away_team, historical_df)
            
            row = {}
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

            all_match_features.append(row)

        if not all_match_features:
            print("Nenhuma partida pôde ser processada.")
            return []

        features_df = pd.DataFrame(all_match_features)
        
        # Garante a mesma ordem de colunas do treino
        features_df = features_df.reindex(columns=feature_columns, fill_value=0)

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
    df = load_dataset()

    if df is not None:
        X, y, feature_columns, live_stats = prepare_features_iterative(df)
        
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
        predictions = predict_next_round(model, scaler, 'next_round.json', live_stats, df, feature_columns)
        if predictions:
            for pred in predictions:
                print(pred)

if __name__ == "__main__":
    main()