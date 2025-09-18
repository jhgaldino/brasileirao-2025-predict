#!/usr/bin/env python3
"""
🚀 Preditor de Futebol - Execução Simples
=========================================

Script ultra-simplificado para gerar predições automaticamente.

Uso:
    python predict.py           # Execução completa (todos os modelos)
    python predict.py --quick   # Execução ultra-rápida (1 modelo sorteado)
"""

import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Preditor de Futebol Automatizado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos disponíveis:
  Completo: Otimiza todos os 7 modelos e escolhe o melhor (~15-30 min)
  Rápido:   Sorteia 1 modelo e otimiza apenas ele (~3-5 min)
        """
    )
    parser.add_argument('--quick', action='store_true', help='Execução ultra-rápida (1 modelo sorteado)')
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Iniciando Predição RÁPIDA...")
        print("🎲 Sorteando 1 modelo para otimização...")
        cmd = ["python", "quick_predict.py"]
    else:
        print("🚀 Iniciando Predição COMPLETA...")
        print("🔍 Otimizando todos os 7 modelos...")
        cmd = ["python", "automated_prediction_pipeline.py"]
    
    # Executar pipeline
    try:
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Predições concluídas com sucesso!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro na execução: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Execução interrompida pelo usuário")
        return 1

if __name__ == "__main__":
    sys.exit(main())