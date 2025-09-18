#!/usr/bin/env python3
"""
⚡ Preditor Rápido de Futebol
============================

Execução ultra-rápida:
1. Sorteia 1 modelo aleatório
2. Otimiza apenas esse modelo (25 iterações)
3. Gera predições
4. Limpa arquivos temporários

Uso:
    python quick_predict.py
"""

import os
import sys
import json
import shutil
import random
import subprocess
from datetime import datetime

class QuickPredictor:
    def __init__(self):
        # Modelos disponíveis
        self.available_models = [
            'smote_gb',
            'adasyn_xgb', 
            'smotetomek_rf',
            'gb_ros',
            'rf_ros',
            'smotetomek_gb',
            'smote_mlp'
        ]
        
        self.temp_dirs = ['optimization_results']
        self.temp_files = ['optimization_comparison.csv']
    
    def print_header(self, message):
        """Imprime cabeçalho formatado"""
        print("\n" + "="*50)
        print(f"⚡ {message}")
        print("="*50)
    
    def run_command(self, command, description):
        """Executa comando e trata erros"""
        print(f"\n🚀 {description}")
        print(f"💻 {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                check=True
            )
            
            if result.stdout:
                print("✅ Resultado:")
                print(result.stdout)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro: {e}")
            if e.stderr:
                print(f"📥 Stderr: {e.stderr}")
            return False
    
    def cleanup_temp_files(self):
        """Remove arquivos temporários"""
        print("\n🧹 Limpando arquivos temporários...")
        
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"✅ Removido: {temp_dir}/")
                except Exception as e:
                    print(f"⚠️ Erro ao remover {temp_dir}: {e}")
        
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"✅ Removido: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Erro ao remover {temp_file}: {e}")
    
    def run_quick_prediction(self):
        """Executa predição rápida"""
        start_time = datetime.now()
        
        self.print_header("PREDIÇÃO RÁPIDA DE FUTEBOL")
        print(f"🕐 Iniciado em: {start_time.strftime('%H:%M:%S')}")
        
        try:
            # Sortear modelo aleatório
            selected_model = random.choice(self.available_models)
            print(f"\n🎲 Modelo sorteado: {selected_model.upper()}")
            
            # Passo 1: Otimização rápida do modelo sorteado
            print(f"\n📋 PASSO 1: Otimização Rápida")
            print("-" * 30)
            
            if not self.run_command(
                f"python hyperparameter_tuning.py --model {selected_model}",
                f"Otimizando {selected_model}"
            ):
                raise Exception("Falha na otimização")
            
            # Passo 2: Aplicar melhores parâmetros
            print(f"\n📋 PASSO 2: Aplicação e Predição")
            print("-" * 30)
            
            if not self.run_command(
                "python apply_best_params.py",
                "Gerando predições"
            ):
                raise Exception("Falha na predição")
            
            # Passo 3: Limpeza
            self.cleanup_temp_files()
            
            # Sucesso
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.print_header("PREDIÇÃO RÁPIDA CONCLUÍDA! 🎉")
            print(f"⏱️ Tempo total: {duration}")
            print(f"🎲 Modelo usado: {selected_model.upper()}")
            print(f"🧹 Arquivos temporários removidos")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            print("\n🔧 Limpeza de emergência...")
            self.cleanup_temp_files()
            return False

def main():
    print("⚡ Iniciando Predição Rápida...")
    
    predictor = QuickPredictor()
    success = predictor.run_quick_prediction()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()