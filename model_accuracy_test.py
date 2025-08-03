# This is a test file for accuracy testing of different balancing methods
# Based on your result, the model with more accuracy is used to predict the results of the next round
# Based on main.py, but with conversion of classes to numbers for it

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# Mapeamento das classes
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
    # Busca os 5 jogos anteriores do time até o índice atual
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
    # Busca confrontos diretos anteriores entre os times
    matches = df.iloc[:current_index]
    h2h = matches[((matches['home_team'] == home_team) & (matches['away_team'] == away_team)) |
                  ((matches['home_team'] == away_team) & (matches['away_team'] == home_team))]
    if h2h.empty:
        return 0.0
    # Proporção de vitórias do mandante nos confrontos diretos
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
    # Novas features
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
        # Novas features
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

def train_model(X, y, balancer='ros', model='xgb'):
    feature_columns = X.columns.tolist()
    X = X.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Escolha do balancer
    if balancer == 'ros':
        print('\nUsando RandomOverSampler')
        sampler = RandomOverSampler(random_state=42)
    elif balancer == 'smote':
        print('\nUsando SMOTE')
        sampler = SMOTE(random_state=42)
    elif balancer == 'smotetomek':
        print('\nUsando SMOTE+TomekLinks')
        sampler = SMOTETomek(random_state=42)
    elif balancer == 'adasyn':
        print('\nUsando ADASYN')
        sampler = ADASYN(random_state=42)
    else:
        print('\nbalancer não reconhecido, usando RandomOverSampler')
        sampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    print('\nDistribuição das classes após balanceamento:')
    print(pd.Series(y_train_resampled).value_counts())
    print(pd.Series(y_train_resampled).value_counts(normalize=True).rename('proporcao'))
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    X_train_resampled = X_train_resampled[feature_columns]
    X_test_scaled = X_test_scaled[feature_columns]
    y_train_res_num = pd.Series(y_train_resampled).map(CLASS_MAP).values
    y_test_num = pd.Series(y_test).map(CLASS_MAP).values
    # models
    if model == 'rf':
        rf = RandomForestClassifier(class_weight={'H': 1.0, 'D': 1.2, 'A': 1.0}, random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
        return rf, scaler, X_test_scaled, y_test, 'rf'
    elif model == 'xgb':
        param_grid_xgb = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
        }
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
        grid_xgb.fit(X_train_resampled, y_train_res_num)
        best_xgb = grid_xgb.best_estimator_
        print(f"\nBest parameters for XGBoost: {grid_xgb.best_params_}")
        return best_xgb, scaler, X_test_scaled, y_test, 'xgb'
    elif model == 'gb':
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train_resampled, y_train_res_num)
        return gb, scaler, X_test_scaled, y_test, 'gb'
    elif model == 'mlp':
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_train_resampled, y_train_res_num)
        return mlp, scaler, X_test_scaled, y_test, 'mlp'
    else:
        raise ValueError('model not recognized')

def predict_with_thresholds(model, X_test_scaled, thresholds):
    """
    Makes predictions using custom thresholds for each class.
    If no threshold is met, it uses the class with the highest probability as a fallback.
    """
    # Get the probabilities for the test set
    y_proba = model.predict_proba(X_test_scaled)
    
    y_pred = []
    for probs in y_proba:
        # Get the probability for each class using the class map
        h_prob = probs[CLASS_MAP['H']]
        d_prob = probs[CLASS_MAP['D']]
        a_prob = probs[CLASS_MAP['A']]

        # Decision logic based on the thresholds
        if d_prob >= thresholds['D'] and d_prob == max(h_prob, d_prob, a_prob):
            y_pred.append('D')
        elif h_prob >= thresholds['H'] and h_prob == max(h_prob, d_prob, a_prob):
            y_pred.append('H')
        elif a_prob >= thresholds['A'] and a_prob == max(h_prob, d_prob, a_prob):
            y_pred.append('A')
        else:
            # Fallback: if no threshold is met, choose the highest probability
            idx = np.argmax(probs)
            y_pred.append(CLASS_MAP_INV[idx])
            
    return y_pred

def main():
    """
    Orchestrates the model testing process:
    1. Loads and prepares the data with advanced features.
    2. Iterates over different balancers and models.
    3. Trains each combination.
    4. Evaluates each model's performance ON THE TEST SET.
    5. Presents a final ranking based on the F1-Score Macro to help choose the best model.
    """
    df = load_data()
    classification_data = load_classification_data()

    if df is not None and classification_data is not None:
        print('Initial class distribution in the dataset:')
        print(df['match_result'].value_counts(normalize=True).rename('proportion'))
        
        X, y = prepare_features(df, classification_data)
        
        balancers = ['ros', 'smote', 'smotetomek', 'adasyn']
        model_names = ['rf', 'xgb', 'gb', 'mlp'] # The list of model names
        results = []

        for balancer_name in balancers:
            for model_name in model_names:
                print(f'\n======================================================')
                print(f'Testing: Balancer [{balancer_name.upper()}] | Model [{model_name.upper()}]')
                print(f'======================================================')
                
                # The train_model function now also returns test data for a fair evaluation
                model, scaler, X_test_scaled, y_test, model_type = train_model(X, y, balancer=balancer_name, model=model_name)
                
                print(f'\n--- Evaluating performance on the TEST SET (unseen data) ---')
                
                # Prediction is made only on the test data
                if model_type == 'rf':
                    y_pred_test = model.predict(X_test_scaled)
                else:
                    thresholds = {'H': 0.40, 'D': 0.30, 'A': 0.30} # You can adjust these
                    y_pred_test = predict_with_thresholds(model, X_test_scaled, thresholds)

                # Metrics are calculated by comparing y_pred_test with y_test
                acc = accuracy_score(y_test, y_pred_test)
                # Added 'zero_division=0' to prevent warnings if a class is not predicted
                report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
                f1_macro = report['macro avg']['f1-score']

                print(f"Test Accuracy: {acc:.4f}")
                print(f"Test F1-Score (Macro Avg): {f1_macro:.4f}")
                print("\nConfusion Matrix (Test):")
                print(confusion_matrix(y_test, y_pred_test, labels=['H', 'D', 'A']))
                print("\nClassification Report (Test):")
                print(classification_report(y_test, y_pred_test, labels=['H', 'D', 'A'], zero_division=0))
                
                # Save the important results for the final comparison
                results.append({
                    'balancer': balancer_name, 
                    'model': model_name, 
                    'accuracy': acc,
                    'f1_macro': f1_macro
                })

        # --- FINAL RANKING ---
        print('\n======================================================')
        print('          FINAL MODEL RANKING')
        print('       (Ordered by F1-Score Macro)')
        print('======================================================')
        
        # Sort the results by F1-Score Macro, from highest to lowest
        sorted_results = sorted(results, key=lambda item: item['f1_macro'], reverse=True)
        
        for r in sorted_results:
            print(f"Balancer: {r['balancer']:<10} | Model: {r['model']:<4} | F1-Macro: {r['f1_macro']:.4f} | Accuracy: {r['accuracy']:.4f}")

        best_combination = sorted_results[0]
        print('\n------------------------------------------------------')
        print(f"🏆 Best combination found: ")
        print(f"   Balancer '{best_combination['balancer']}' with model '{best_combination['model']}'")
        print(f"   F1-Score: {best_combination['f1_macro']:.4f}")
        print('------------------------------------------------------')


if __name__ == "__main__":
    main()