import pandas as pd
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

# Import samplers
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore', category=UserWarning)

# Mapeamento das classes
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}

# =============================================================================
# FUNÇÕES DE DADOS (Reutilizadas)
# =============================================================================
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

def prepare_features(df, classification_data):
    df['points_diff'] = 0
    df['position_diff'] = 0
    df['home_strength'] = 0.0
    df['away_strength'] = 0.0
    df['strength_diff'] = 0.0

    for idx, row in df.iterrows():
        home_stats = get_team_stats_from_table(row['home_team'], classification_data)
        away_stats = get_team_stats_from_table(row['away_team'], classification_data)
        
        home_points = home_stats.get('pts', 0)
        away_points = away_stats.get('pts', 0)
        home_pos = home_stats.get('posicao', 20)
        away_pos = away_stats.get('posicao', 20)
        home_played = home_stats.get('pj', 1)
        away_played = away_stats.get('pj', 1)

        df.loc[idx, 'points_diff'] = home_points - away_points
        df.loc[idx, 'position_diff'] = away_pos - home_pos
        df.loc[idx, 'home_strength'] = home_points / max(1, home_played)
        df.loc[idx, 'away_strength'] = away_points / max(1, away_played)
        df.loc[idx, 'strength_diff'] = df.loc[idx, 'home_strength'] - df.loc[idx, 'away_strength']

    feature_columns = [
        'home_possession', 'away_possession', 'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
        'home_corners', 'away_corners', 'home_passes', 'away_passes', 'home_pass_accuracy', 'away_pass_accuracy',
        'home_fouls', 'away_fouls', 'home_yellow_cards', 'away_yellow_cards', 'home_offsides', 'away_offsides',
        'home_red_cards', 'away_red_cards', 'home_crosses', 'away_crosses', 'home_cross_accuracy', 'away_cross_accuracy',
        'points_diff', 'position_diff', 'home_strength', 'away_strength', 'strength_diff'
    ]
    
    X = df[feature_columns].fillna(0)
    y = df['match_result']
    
    return X, y

# =============================================================================
# CONFIGURAÇÃO CENTRAL DE OTIMIZAÇÃO
# =============================================================================

# Define os estimadores e suas grades de parâmetros
tuning_config = {
    'adasyn_xgb': {
        'estimator': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'sampler': ADASYN(random_state=42),
        'param_grid': {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    },
    'smotetomek_rf': {
        'estimator': RandomForestClassifier(random_state=42),
        'sampler': SMOTETomek(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'gb_ros': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'sampler': RandomOverSampler(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'rf_ros': {
        'estimator': RandomForestClassifier(random_state=42),
        'sampler': RandomOverSampler(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}

# =============================================================================
# FUNÇÃO PRINCIPAL DE OTIMIZAÇÃO
# =============================================================================

def main(models_to_tune):
    """
    Orquestra o processo de otimização de hiperparâmetros para os modelos especificados.
    """
    print("Carregando e preparando os dados...")
    df = load_data()
    classification_data = load_classification_data()

    if df is None or classification_data is None:
        print("Não foi possível carregar os dados. Abortando.")
        return
        
    X, y = prepare_features(df, classification_data)
    y_numeric = y.map(CLASS_MAP)

    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Itera sobre os modelos a serem otimizados
    for model_name in models_to_tune:
        if model_name not in tuning_config:
            print(f"Modelo '{model_name}' não reconhecido. Pulando.")
            continue

        config = tuning_config[model_name]
        estimator = config['estimator']
        sampler = config['sampler']
        param_grid = config['param_grid']

        print(f"\n{'='*60}")
        print(f"INICIANDO OTIMIZAÇÃO PARA O MODELO: {model_name.upper()}")
        print(f"{'='*60}")

        # Balanceamento dos dados
        print(f"Aplicando {sampler.__class__.__name__} para balanceamento...")
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
        print("Balanceamento concluído.")

        # Configuração do GridSearchCV
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            verbose=2
        )

        print("\nIniciando a busca pelos melhores hiperparâmetros com GridSearchCV...")
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Apresentação dos resultados
        print(f"\n--- RESULTADO PARA {model_name.upper()} ---")
        print(f"Melhor pontuação (F1-Macro): {grid_search.best_score_:.4f}")
        print("\nMelhores Hiperparâmetros Encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"- {param}: {value}")
        
        # Avaliação no conjunto de teste
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(X_test_scaled)
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        print(f"\nPontuação F1-Macro no conjunto de teste: {f1_test:.4f}")
        print(f"{'-'*40}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otimizador de Hiperparâmetros para Modelos de Classificação")
    parser.add_argument(
        '--model',
        nargs='+',
        choices=list(tuning_config.keys()),
        default=list(tuning_config.keys()),
        help=f"Especifique um ou mais modelos para otimizar. Padrão: todos. Opções: {', '.join(tuning_config.keys())}"
    )
    args = parser.parse_args()
    
    main(args.model)