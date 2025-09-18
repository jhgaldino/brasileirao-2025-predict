import json
import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from common.data_utils import load_dataset, prepare_features_iterative, get_last5_form, get_h2h_history

# class labels
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}

def load_best_smote_gb_params():
    """
    Carrega os melhores parâmetros para o modelo SMOTE + Gradient Boosting.
    """
    results_dir = "optimization_results"
    if not os.path.exists(results_dir):
        print("Diretório de resultados não encontrado. Usando parâmetros padrão.")
        return None
    
    # Procura por arquivos do modelo smote_gb
    smote_gb_files = glob(f"{results_dir}/smote_gb_best_params_*.json")
    
    if not smote_gb_files:
        print("Nenhum resultado de otimização encontrado para smote_gb. Usando parâmetros padrão.")
        return None
    
    # Pega o arquivo mais recente
    latest_file = max(smote_gb_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Carregando parâmetros otimizados de: {latest_file}")
        print(f"CV Score: {data['cv_score']:.4f}")
        print(f"Test Score: {data['test_score']:.4f}")
        
        return data['best_params']
    
    except Exception as e:
        print(f"Erro ao carregar parâmetros: {e}")
        return None

def train_optimized_gb_smote_model(X, y, best_params=None):
    """
    Treina o modelo Gradient Boosting com SMOTE usando os melhores parâmetros encontrados.
    """
    print("Treinando o modelo Gradient Boosting com SMOTE...")

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Mapeia os resultados (H, D, A) para números (0, 1, 2)
    y_resampled_num = pd.Series(y_resampled).map(CLASS_MAP).values

    # Configuração do modelo com parâmetros otimizados
    if best_params:
        print("Usando parâmetros otimizados:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        model = GradientBoostingClassifier(
            random_state=42,
            **best_params
        )
    else:
        print("Usando parâmetros padrão (otimizados manualmente):")
        default_params = {
            'n_estimators': 165,
            'learning_rate': 0.09015555039236096,
            'max_depth': 10,
            'subsample': 0.756527587529629
        }
        for param, value in default_params.items():
            print(f"  {param}: {value}")
        
        model = GradientBoostingClassifier(
            random_state=42,
            **default_params
        )
    
    model.fit(X_resampled, y_resampled_num)
    
    print("Modelo treinado com sucesso.")
    return model, scaler

def predict_next_round(model, scaler, next_round_file, live_stats, live_ewma_stats, historical_df, feature_columns):
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
    stat_features = list(live_ewma_stats[list(live_ewma_stats.keys())[0]].keys())

    for match in next_round_data.get('partidas', []):
        home_team = match.get('mandante')
        away_team = match.get('visitante')

        if not home_team or not away_team:
            print(f"Skipping match with missing team names: {match}")
            continue

        if home_team not in live_stats or away_team not in live_stats:
            print(f"Skipping match because a team is not in live_stats: {home_team} or {away_team}")
            continue

        row = {}
        # Features de EWMA
        for stat in stat_features:
            row[f'home_{stat}_ewma'] = live_ewma_stats[home_team][stat]
            row[f'away_{stat}_ewma'] = live_ewma_stats[away_team][stat]

        # Features de classificação e outras
        home_s = live_stats[home_team]
        away_s = live_stats[away_team]
        home_played = max(1, home_s['pj'])
        away_played = max(1, away_s['pj'])

        row['home_last5_form'] = get_last5_form(home_team, historical_df, len(historical_df))
        row['away_last5_form'] = get_last5_form(away_team, historical_df, len(historical_df))
        row['strength_diff'] = (home_s['pts'] / home_played) - (away_s['pts'] / away_played)
        row['form_momentum'] = row['home_last5_form'] - row['away_last5_form']
        row['h2h_home_win_rate'] = get_h2h_history(home_team, away_team, historical_df, len(historical_df))

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
    Função principal que usa os melhores parâmetros encontrados na otimização.
    """
    print("="*60)
    print("MODELO COM PARÂMETROS OTIMIZADOS")
    print("="*60)
    
    # 1. Carregar melhores parâmetros
    best_params = load_best_smote_gb_params()
    
    # 2. Carregar dados históricos
    df = load_dataset()

    if df is None:
        print("It was not possible to load the data.")
        return
        
    # 3. Preparar features para treinamento
    X, y, feature_columns, live_stats, live_ewma_stats = prepare_features_iterative(df)
    
    # 4. Treinar o modelo GB + SMOTE com parâmetros otimizados
    model, scaler = train_optimized_gb_smote_model(X, y, best_params)
    
    # 5. Fazer predições para a próxima rodada
    predict_next_round(model, scaler, 'next_round.json', live_stats, live_ewma_stats, df, feature_columns)

if __name__ == "__main__":
    main()