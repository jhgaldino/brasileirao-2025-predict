import json

def calcular_medias():
    with open('dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Initialize accumulators
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
    total_jogos = 0

    def get_stat_value(stats, key):
        if key not in stats:
            return 0
        value = stats[key]
        if isinstance(value, str) and value.endswith('%'):
            return float(value.strip('%'))
        return float(value) if value else 0

    # Sum all statistics across all rounds
    for rodada in data['rodadas']:
        for partida in rodada['partidas']:
            if 'estatisticas' not in partida:
                continue
                
            total_jogos += 1
            estatisticas = partida['estatisticas']
            mandante = partida['partida']['mandante']
            visitante = partida['partida']['visitante']

            # Get the statistics for each team using the team names from the match data
            mandante_stats = None
            visitante_stats = None
            
            # Find the correct statistics for each team
            for team_name, team_stats in estatisticas.items():
                if team_name == mandante:
                    mandante_stats = team_stats
                elif team_name == visitante:
                    visitante_stats = team_stats
            
            if not mandante_stats or not visitante_stats:
                continue

            # Home team statistics
            for stat in stats_mandante:
                stats_mandante[stat] += get_stat_value(mandante_stats, stat)
            
            # Away team statistics
            for stat in stats_visitante:
                stats_visitante[stat] += get_stat_value(visitante_stats, stat)
    
    # Calculate averages
    for stat in stats_mandante:
        stats_mandante[stat] = round(stats_mandante[stat] / total_jogos, 2)
        stats_visitante[stat] = round(stats_visitante[stat] / total_jogos, 2)
    
    return {
        'home_possession': stats_mandante['posse_de_bola'],
        'away_possession': stats_visitante['posse_de_bola'],
        'home_shots': stats_mandante['chutes'],
        'away_shots': stats_visitante['chutes'],
        'home_shots_target': stats_mandante['chutes_a_gol'],
        'away_shots_target': stats_visitante['chutes_a_gol'],
        'home_corners': stats_mandante['escanteios'],
        'away_corners': stats_visitante['escanteios'],
        'home_passes': stats_mandante['passes'],
        'away_passes': stats_visitante['passes'],
        'home_pass_accuracy': stats_mandante['precisao_de_passe'],
        'away_pass_accuracy': stats_visitante['precisao_de_passe'],
        'home_fouls': stats_mandante['faltas'],
        'away_fouls': stats_visitante['faltas'],
        'home_yellow_cards': stats_mandante['cartoes_amarelos'],
        'away_yellow_cards': stats_visitante['cartoes_amarelos'],
        'home_offsides': stats_mandante['impedimentos'],
        'away_offsides': stats_visitante['impedimentos'],
        'home_red_cards': stats_mandante['cartoes_vermelhos'],
        'away_red_cards': stats_visitante['cartoes_vermelhos'],
        'home_crosses': stats_mandante['cruzamentos'],
        'away_crosses': stats_visitante['cruzamentos'],
        'home_cross_accuracy': stats_mandante['precisao_cruzamento'],
        'away_cross_accuracy': stats_visitante['precisao_cruzamento']
    }

if __name__ == '__main__':
    stats = calcular_medias()
    print("\nMédias dos Times Mandantes:")
    for key, value in stats.items():
        if key.startswith('home_'):
            print(f"{key}: {value}")
    
    print("\nMédias dos Times Visitantes:")
    for key, value in stats.items():
        if key.startswith('away_'):
            print(f"{key}: {value}")