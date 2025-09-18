#!/usr/bin/env python3
"""
Pipeline Automatizado de Predição de Futebol
============================================

Este script executa todo o processo de otimização e predição automaticamente:
1. Otimização Bayesiana de todos os modelos
2. Comparação e seleção do melhor modelo
3. Aplicação dos melhores parâmetros
4. Geração das predições finais
5. Limpeza automática dos arquivos temporários

Uso:
    python automated_prediction_pipeline.py [--quick] [--keep-files]
    
Argumentos:
    --quick: Execução rápida (menos iterações)
    --keep-files: Mantém arquivos temporários (para debug)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class AutomatedPredictionPipeline:
    def __init__(self, quick_mode=False, keep_files=False):
        self.quick_mode = quick_mode
        self.keep_files = keep_files
        self.temp_dirs = ['optimization_results']
        self.temp_files = ['optimization_comparison.csv']
        
    def print_header(self, message):
        """Imprime cabeçalho formatado"""
        print("\n" + "="*60)
        print(f"🚀 {message}")
        print("="*60)
        
    def print_step(self, step, message):
        """Imprime passo atual"""
        print(f"\n📋 PASSO {step}: {message}")
        print("-" * 40)
        
    def run_command(self, command, description):
        """Executa comando e trata erros"""
        print(f"⚡ Executando: {description}")
        print(f"💻 Comando: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                check=True
            )
            
            if result.stdout:
                print("✅ Saída:")
                print(result.stdout)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao executar: {description}")
            print(f"💥 Código de erro: {e.returncode}")
            if e.stdout:
                print(f"📤 Stdout: {e.stdout}")
            if e.stderr:
                print(f"📥 Stderr: {e.stderr}")
            return False
    
    def cleanup_temp_files(self):
        """Remove arquivos e diretórios temporários"""
        if self.keep_files:
            print("\n🗂️ Mantendo arquivos temporários (--keep-files ativado)")
            return
            
        print("\n🧹 Limpando arquivos temporários...")
        
        # Remove diretórios temporários
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"✅ Removido diretório: {temp_dir}/")
                except Exception as e:
                    print(f"⚠️ Erro ao remover {temp_dir}: {e}")
        
        # Remove arquivos temporários
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"✅ Removido arquivo: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Erro ao remover {temp_file}: {e}")
    
    def check_dependencies(self):
        """Verifica se os arquivos necessários existem"""
        required_files = [
            'dataset.json',
            'next_round.json',
            'hyperparameter_tuning.py',
            'compare_optimization_results.py',
            'apply_best_params.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ Arquivos necessários não encontrados:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ Todos os arquivos necessários encontrados")
        return True
    
    def run_pipeline(self):
        """Executa o pipeline completo"""
        start_time = datetime.now()
        
        self.print_header("PIPELINE AUTOMATIZADO DE PREDIÇÃO")
        print(f"🕐 Iniciado em: {start_time.strftime('%H:%M:%S')}")
        print(f"⚡ Modo: {'Rápido' if self.quick_mode else 'Completo'}")
        print(f"🗂️ Arquivos temporários: {'Mantidos' if self.keep_files else 'Removidos'}")
        
        # Verificar dependências
        self.print_step(0, "Verificando Dependências")
        if not self.check_dependencies():
            print("❌ Pipeline interrompido devido a arquivos faltantes")
            return False
        
        try:
            # Passo 1: Otimização de Hiperparâmetros
            self.print_step(1, "Otimização Bayesiana de Hiperparâmetros")
            
            # Executar diretamente o hyperparameter_tuning.py (executa todos por padrão)
            if not self.run_command(
                "python hyperparameter_tuning.py",
                "Otimização de todos os modelos"
            ):
                raise Exception("Falha na otimização de hiperparâmetros")
            
            # Passo 2: Comparação de Resultados
            self.print_step(2, "Comparação e Ranking dos Modelos")
            
            if not self.run_command(
                "python compare_optimization_results.py",
                "Comparação de performance dos modelos"
            ):
                raise Exception("Falha na comparação de resultados")
            
            # Passo 3: Aplicação dos Melhores Parâmetros
            self.print_step(3, "Aplicação dos Melhores Parâmetros")
            
            if not self.run_command(
                "python apply_best_params.py",
                "Aplicação dos parâmetros otimizados e predição"
            ):
                raise Exception("Falha na aplicação dos melhores parâmetros")
            
            # Passo 4: Mostrar Resumo Final
            self.print_step(4, "Resumo Final")
            self.show_final_summary()
            
            # Passo 5: Limpeza
            self.print_step(5, "Limpeza de Arquivos Temporários")
            self.cleanup_temp_files()
            
            # Sucesso
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.print_header("PIPELINE CONCLUÍDO COM SUCESSO! 🎉")
            print(f"⏱️ Tempo total: {duration}")
            print(f"🏆 Predições geradas com o melhor modelo otimizado")
            print(f"🧹 Arquivos temporários {'mantidos' if self.keep_files else 'removidos'}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERRO NO PIPELINE: {e}")
            print("\n🔧 Tentando limpeza de emergência...")
            self.cleanup_temp_files()
            return False
    
    def show_final_summary(self):
        """Mostra resumo final dos resultados"""
        try:
            # Tentar ler o arquivo de comparação se existir
            if os.path.exists('optimization_comparison.csv'):
                print("📊 Resumo da Otimização:")
                with open('optimization_comparison.csv', 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Header + pelo menos 1 linha
                        print("🏆 Top 3 Modelos:")
                        for i, line in enumerate(lines[1:4], 1):  # Skip header, show top 3
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                print(f"   {i}. {parts[0]} - Score: {parts[1]}")
            
            print("\n🎯 Predições geradas para a próxima rodada!")
            print("📁 Verifique a saída acima para os resultados detalhados")
            
        except Exception as e:
            print(f"⚠️ Não foi possível mostrar resumo detalhado: {e}")
            print("✅ Pipeline executado, verifique os logs acima")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Automatizado de Predição de Futebol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python automated_prediction_pipeline.py                 # Execução completa
  python automated_prediction_pipeline.py --quick         # Execução rápida
  python automated_prediction_pipeline.py --keep-files    # Manter arquivos temporários
  python automated_prediction_pipeline.py --quick --keep-files  # Rápido + manter arquivos
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Execução rápida com menos iterações (para testes)'
    )
    
    parser.add_argument(
        '--keep-files', 
        action='store_true',
        help='Manter arquivos temporários após execução (para debug)'
    )
    
    args = parser.parse_args()
    
    # Criar e executar pipeline
    pipeline = AutomatedPredictionPipeline(
        quick_mode=args.quick,
        keep_files=args.keep_files
    )
    
    success = pipeline.run_pipeline()
    
    # Exit code baseado no sucesso
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()