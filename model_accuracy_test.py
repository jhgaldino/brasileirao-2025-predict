import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

from common.data_utils import load_dataset, prepare_features_iterative, CLASS_MAP

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURAÇÃO DE MODELOS E HIPERPARÂMETROS
# =============================================================================

model_config = {
    'rf': {
        'estimator': RandomForestClassifier(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 11),
            'min_samples_leaf': Integer(1, 5)
        }
    },
    'xgb': {
        'estimator': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'search_spaces': {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'subsample': Real(0.7, 1.0, 'uniform'),
            'colsample_bytree': Real(0.7, 1.0, 'uniform')
        }
    },
    'gb': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'search_spaces': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.7, 1.0, 'uniform')
        }
    }
}

balancer_config = {
    'ros': RandomOverSampler(random_state=42),
    'smote': SMOTE(random_state=42),
    'smotetomek': SMOTETomek(random_state=42),
    'adasyn': ADASYN(random_state=42)
}

# =============================================================================
# FUNÇÃO PRINCIPAL DE TESTE DE ACURÁCIA
# =============================================================================

def main():
    """
    Orchests the hyperparameter tuning and model evaluation process.
    
    """
    df = load_dataset()
    if df is None:
        print("Não foi possível carregar os dados. Abortando.")
        return

    print('Distribuição inicial de classes no dataset:')
    print(df['match_result'].value_counts(normalize=True).rename('proportion'))
    
    X, y, _, _, _ = prepare_features_iterative(df)
    y_numeric = y.map(CLASS_MAP)

    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for balancer_name, sampler in balancer_config.items():
        for model_name, config in model_config.items():
            print(f'\n{'='*60}')
            print(f'TESTANDO: Balanceador [{balancer_name.upper()}] | Modelo [{model_name.upper()}]')
            print(f'{'='*60}')

            print(f"Aplicando {sampler.__class__.__name__} para balanceamento...")
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            print("Balanceamento concluído.")

            bayes_search = BayesSearchCV(
                estimator=config['estimator'],
                search_spaces=config['search_spaces'],
                n_iter=20,  # Reduzido para um teste mais rápido, aumente para uma busca mais exaustiva
                scoring='f1_macro',
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            print("\nIniciando BayesSearchCV...")
            bayes_search.fit(X_train_resampled, y_train_resampled)

            best_model = bayes_search.best_estimator_
            y_pred_test = best_model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred_test)
            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            f1_macro = report['macro avg']['f1-score']

            print(f"\n--- AVALIAÇÃO NO CONJUNTO DE TESTE ---")
            print(f"Melhores Hiperparâmetros: {bayes_search.best_params_}")
            print(f"Acurácia no Teste: {acc:.4f}")
            print(f"F1-Score (Macro Avg) no Teste: {f1_macro:.4f}")
            print("\nRelatório de Classificação (Teste):")
            print(classification_report(y_test, y_pred_test, labels=list(CLASS_MAP.values()), target_names=list(CLASS_MAP.keys()), zero_division=0))

            results.append({
                'balancer': balancer_name,
                'model': model_name,
                'accuracy': acc,
                'f1_macro': f1_macro,
                'best_params': bayes_search.best_params_
            })

    # --- RANKING FINAL ---
    print(f'\n{'='*60}')
    print('          RANKING FINAL DOS MODELOS')
    print('       (Ordenado por F1-Score Macro)')
    print(f'{'='*60}')

    sorted_results = sorted(results, key=lambda item: item['f1_macro'], reverse=True)

    for r in sorted_results:
        print(f"Balanceador: {r['balancer']:<10} | Modelo: {r['model']:<4} | F1-Macro: {r['f1_macro']:.4f} | Acurácia: {r['accuracy']:.4f}")

    best_combination = sorted_results[0]
    print(f'\n{'-'*60}')
    print(f"🏆 Melhor Combinação Encontrada: ")
    print(f"   Balanceador '{best_combination['balancer']}' com modelo '{best_combination['model']}'")
    print(f"   F1-Score: {best_combination['f1_macro']:.4f}")
    print(f"   Melhores Parâmetros: {best_combination['best_params']}")
    print(f'{'-'*60}')

if __name__ == "__main__":
    main()
