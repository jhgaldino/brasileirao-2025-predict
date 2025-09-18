import json
import os
import pandas as pd
from glob import glob

def load_optimization_results():
    """
    Carrega todos os resultados de otimização salvos.
    """
    results_dir = "optimization_results"
    if not os.path.exists(results_dir):
        print(f"Diretório {results_dir} não encontrado.")
        return None
    
    json_files = glob(f"{results_dir}/*.json")
    if not json_files:
        print("Nenhum arquivo de resultado encontrado.")
        return None
    
    results = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
    
    return results

def create_comparison_table(results):
    """
    Cria uma tabela comparativa dos resultados.
    """
    comparison_data = []
    
    for result in results:
        row = {
            'Model': result['model_name'],
            'CV_Score': result['cv_score'],
            'Test_Score': result['test_score'],
            'Timestamp': result['timestamp']
        }
        
        # Adiciona os melhores parâmetros como colunas separadas
        for param, value in result['best_params'].items():
            row[f'best_{param}'] = value
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('Test_Score', ascending=False)

def print_best_models(df, top_n=3):
    """
    Imprime os melhores modelos encontrados.
    """
    print(f"\n{'='*60}")
    print(f"TOP {top_n} MODELOS COM MELHOR PERFORMANCE")
    print(f"{'='*60}")
    
    top_models = df.head(top_n)
    
    for idx, (_, row) in enumerate(top_models.iterrows(), 1):
        print(f"\n{idx}º Lugar: {row['Model']}")
        print(f"   CV Score: {row['CV_Score']:.4f}")
        print(f"   Test Score: {row['Test_Score']:.4f}")
        print(f"   Timestamp: {row['Timestamp']}")
        
        # Mostra os parâmetros principais
        param_cols = [col for col in row.index if col.startswith('best_')]
        if param_cols:
            print("   Melhores Parâmetros:")
            for param_col in param_cols:
                param_name = param_col.replace('best_', '')
                print(f"     {param_name}: {row[param_col]}")

def main():
    """
    Função principal para comparar resultados de otimização.
    """
    print("Carregando resultados de otimização...")
    results = load_optimization_results()
    
    if not results:
        return
    
    print(f"Encontrados {len(results)} resultados.")
    
    # Criar tabela comparativa
    df = create_comparison_table(results)
    
    # Salvar tabela completa
    df.to_csv('optimization_comparison.csv', index=False)
    print("\nTabela comparativa salva em: optimization_comparison.csv")
    
    # Mostrar resumo
    print(f"\n{'='*60}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*60}")
    print(f"Modelos testados: {len(df)}")
    print(f"Melhor CV Score: {df['CV_Score'].max():.4f}")
    print(f"Melhor Test Score: {df['Test_Score'].max():.4f}")
    print(f"Score médio (Test): {df['Test_Score'].mean():.4f}")
    
    # Mostrar top modelos
    print_best_models(df)
    
    # Mostrar tabela resumida
    print(f"\n{'='*60}")
    print("TABELA RESUMIDA")
    print(f"{'='*60}")
    summary_cols = ['Model', 'CV_Score', 'Test_Score', 'Timestamp']
    print(df[summary_cols].to_string(index=False))

if __name__ == "__main__":
    main()