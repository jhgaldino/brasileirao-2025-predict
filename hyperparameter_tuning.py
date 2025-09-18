import pandas as pd
import numpy as np
import json
import argparse
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import warnings

# Import samplers
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

# Import models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Import from common utils
from common.data_utils import load_dataset, prepare_features_iterative, CLASS_MAP

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURAÇÃO CENTRAL DE OTIMIZAÇÃO
# =============================================================================

# Define os estimadores e suas grades de parâmetros para BayesSearchCV
tuning_config = {
    'adasyn_xgb': {
        'estimator': XGBClassifier(
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='mlogloss'
        ),
        'sampler': ADASYN(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 800),
            'max_depth': Integer(3, 12),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'subsample': Real(0.7, 1.0, 'uniform'),
            'colsample_bytree': Real(0.7, 1.0, 'uniform'),
            'reg_alpha': Real(0.0, 1.0, 'uniform'),
            'reg_lambda': Real(0.0, 1.0, 'uniform')
        }
    },
    'smotetomek_rf': {
        'estimator': RandomForestClassifier(random_state=42),
        'sampler': SMOTETomek(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 11),
            'min_samples_leaf': Integer(1, 5)
        }
    },
    'gb_ros': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'sampler': RandomOverSampler(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.7, 1.0, 'uniform')
        }
    },
    'rf_ros': {
        'estimator': RandomForestClassifier(random_state=42),
        'sampler': RandomOverSampler(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 11),
            'min_samples_leaf': Integer(1, 5)
        }
    },
    'smotetomek_gb': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'sampler': SMOTETomek(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.7, 1.0, 'uniform')
        }
    },
    'smote_gb': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'sampler': SMOTE(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 800),
            'learning_rate': Real(0.01, 0.25, 'log-uniform'),
            'max_depth': Integer(3, 12),
            'subsample': Real(0.7, 1.0, 'uniform'),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10)
        }
    },
    'smote_mlp': {
        'estimator': MLPClassifier(random_state=42, max_iter=1000),
        'sampler': SMOTE(random_state=42),
        'search_spaces': {
            'hidden_layer_sizes': Categorical([50, 100, 150]),
            'activation': Categorical(['relu', 'tanh']),
            'solver': Categorical(['adam', 'lbfgs']),
            'alpha': Real(0.0001, 0.01, 'log-uniform'),
            'learning_rate_init': Real(0.001, 0.1, 'log-uniform')
        }
    }
}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def save_best_params(model_name, best_params, best_score, test_score, classification_rep):
    """
    Salva os melhores parâmetros encontrados em um arquivo JSON.
    """
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/{model_name}_best_params_{timestamp}.json"
    
    result_data = {
        "model_name": model_name,
        "timestamp": timestamp,
        "best_params": best_params,
        "cv_score": best_score,
        "test_score": test_score,
        "classification_report": classification_rep
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados salvos em: {filename}")

# =============================================================================
# FUNÇÃO PRINCIPAL DE OTIMIZAÇÃO
# =============================================================================

def main(models_to_tune):
    """
    Orquestra o processo de otimização de hiperparâmetros para os modelos especificados.
    """
    print("Carregando e preparando os dados...")
    df = load_dataset()

    if df is None:
        print("Não foi possível carregar os dados. Abortando.")
        return
        
    X, y, _, _, _ = prepare_features_iterative(df)
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
        search_spaces = config['search_spaces']

        print(f"\n{'='*60}")
        print(f"INICIANDO OTIMIZAÇÃO PARA O MODELO: {model_name.upper()}")
        print(f"{'='*60}")

        # Balanceamento dos dados
        print(f"Aplicando {sampler.__class__.__name__} para balanceamento...")
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
        print("Balanceamento concluído.")

        # Configuração do BayesSearchCV
        # Usar menos iterações quando otimizando apenas 1 modelo
        n_iterations = 25 if len(models_to_tune) == 1 else 75
        
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=n_iterations,
            scoring='f1_macro',
            cv=5,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        print("\nIniciando a busca pelos melhores hiperparâmetros com BayesSearchCV...")
        bayes_search.fit(X_train_resampled, y_train_resampled)

        # Apresentação dos resultados
        print(f"\n--- RESULTADO PARA {model_name.upper()} ---")
        print(f"Melhor pontuação (F1-Macro): {bayes_search.best_score_:.4f}")
        print("\nMelhores Hiperparâmetros Encontrados:")
        for param, value in bayes_search.best_params_.items():
            print(f"- {param}: {value}")
        
        # Avaliação no conjunto de teste
        best_model = bayes_search.best_estimator_
        y_pred_test = best_model.predict(X_test_scaled)
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        
        # Relatório de classificação detalhado
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        print(f"\nPontuação F1-Macro no conjunto de teste: {f1_test:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred_test))
        
        # Salvar resultados
        save_best_params(
            model_name, 
            bayes_search.best_params_, 
            bayes_search.best_score_, 
            f1_test,
            class_report
        )
        
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
