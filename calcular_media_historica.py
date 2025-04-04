import json

def calcular_medias():
    with open('dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Inicializar acumuladores
    stats_mandante = {
        "chutes": 0,
        "chutes_a_gol": 0,
        "posse_de_bola": 0,
        "passes": 0,
        "precisao_de_passe": 0,
        "faltas": 0,
        "cartoes_amarelos": 0,
        "cartoes_vermelhos": 0,
        "impedimentos": 0,
        "escanteios": 0,
        "cruzamentos": 0,
        "precisao_cruzamento": 0
    }
    
    stats_visitante = stats_mandante.copy()
    
    # Contador de jogos
    total_jogos = len(data['partidas'])
    
    # Somar todas as estatísticas
    for partida in data['partidas']:
        estatisticas = partida['estatisticas']
        mandante = list(estatisticas.keys())[0]
        visitante = list(estatisticas.keys())[1]
        
        def get_stat_value(stats, key):
            value = stats.get(key, 0)
            if isinstance(value, str) and value.endswith('%'):
                return float(value.strip('%'))
            return value
        
        # Estatísticas do mandante
        stats_mandante['chutes'] += get_stat_value(estatisticas[mandante], 'chutes')
        stats_mandante['chutes_a_gol'] += get_stat_value(estatisticas[mandante], 'chutes_a_gol')
        stats_mandante['posse_de_bola'] += get_stat_value(estatisticas[mandante], 'posse_de_bola')
        stats_mandante['passes'] += get_stat_value(estatisticas[mandante], 'passes')
        stats_mandante['precisao_de_passe'] += get_stat_value(estatisticas[mandante], 'precisao_de_passe')
        stats_mandante['faltas'] += get_stat_value(estatisticas[mandante], 'faltas')
        stats_mandante['cartoes_amarelos'] += get_stat_value(estatisticas[mandante], 'cartoes_amarelos')
        stats_mandante['cartoes_vermelhos'] += get_stat_value(estatisticas[mandante], 'cartoes_vermelhos')
        stats_mandante['impedimentos'] += get_stat_value(estatisticas[mandante], 'impedimentos')
        stats_mandante['escanteios'] += get_stat_value(estatisticas[mandante], 'escanteios')
        stats_mandante['cruzamentos'] += get_stat_value(estatisticas[mandante], 'cruzamentos')
        stats_mandante['precisao_cruzamento'] += get_stat_value(estatisticas[mandante], 'precisao_cruzamento')
        
        # Estatísticas do visitante
        stats_visitante['chutes'] += get_stat_value(estatisticas[visitante], 'chutes')
        stats_visitante['chutes_a_gol'] += get_stat_value(estatisticas[visitante], 'chutes_a_gol')
        stats_visitante['posse_de_bola'] += get_stat_value(estatisticas[visitante], 'posse_de_bola')
        stats_visitante['passes'] += get_stat_value(estatisticas[visitante], 'passes')
        stats_visitante['precisao_de_passe'] += get_stat_value(estatisticas[visitante], 'precisao_de_passe')
        stats_visitante['faltas'] += get_stat_value(estatisticas[visitante], 'faltas')
        stats_visitante['cartoes_amarelos'] += get_stat_value(estatisticas[visitante], 'cartoes_amarelos')
        stats_visitante['cartoes_vermelhos'] += get_stat_value(estatisticas[visitante], 'cartoes_vermelhos')
        stats_visitante['impedimentos'] += get_stat_value(estatisticas[visitante], 'impedimentos')
        stats_visitante['escanteios'] += get_stat_value(estatisticas[visitante], 'escanteios')
        stats_visitante['cruzamentos'] += get_stat_value(estatisticas[visitante], 'cruzamentos')
        stats_visitante['precisao_cruzamento'] += get_stat_value(estatisticas[visitante], 'precisao_cruzamento')
    
    # Calcular médias
    for stat in stats_mandante:
        stats_mandante[stat] = round(stats_mandante[stat] / total_jogos, 2)
        stats_visitante[stat] = round(stats_visitante[stat] / total_jogos, 2)
    
    print("\nMédias dos Times Mandantes:")
    for stat, valor in stats_mandante.items():
        print(f"{stat}: {valor}")
    
    print("\nMédias dos Times Visitantes:")
    for stat, valor in stats_visitante.items():
        print(f"{stat}: {valor}")

calcular_medias()