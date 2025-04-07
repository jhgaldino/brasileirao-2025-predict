# BR-2025-PREDICT

## Previsão de Resultados do Brasileirão 2025

### Descrição
Este projeto utiliza Machine Learning para prever resultados das partidas do Campeonato Brasileiro 2025, analisando estatísticas históricas e desempenho dos times.

### Funcionalidades
- Previsão de resultados (Vitória/Empate/Derrota)
- Análise de estatísticas completas:
  - Posse de bola
  - Finalizações (Total e no gol)
  - Precisão de passes
  - Escanteios
  - Cruzamentos
  - Faltas
  - Cartões (Amarelos/Vermelhos)
  - E muito mais...
- Integração com dados da classificação
- Análise de desempenho histórico

### Fonte dos Dados
- Estatísticas de partidas do Google Sports
- Dados complementares do [FlashScore](https://www.flashscore.com.br)
- Classificação e métricas de desempenho dos times

### Detalhes Técnicos
- Classificador Random Forest
- Features combinando estatísticas de jogo e métricas de desempenho
- Treinamento com dados históricos de partidas
- Atualização contínua com novos dados

### Dataset
O arquivo `dataset.json` contém:
- Resultados das partidas
- Estatísticas detalhadas dos jogos
- Métricas de desempenho dos times
- Atualização após cada rodada

### Como Usar
1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt

