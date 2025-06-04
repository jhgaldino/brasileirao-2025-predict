import json

def calcular_medias_por_time(): # Nome da função alterado para clareza
    with open('dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Dicionário para armazenar as estatísticas acumuladas por time
    team_stats_accumulator = {}

    def get_stat_value(stats, key, default_value=0):
        if key not in stats:
            return default_value
        value = stats[key]
        if isinstance(value, str) and value.endswith('%'):
            return float(value.strip('%'))
        return float(value) if value else default_value

    # Itera por todas as rodadas e partidas
    for rodada in data['rodadas']:
        for partida in rodada['partidas']:
            if 'estatisticas' not in partida:
                continue

            estatisticas = partida['estatisticas']
            mandante = partida['partida']['mandante']
            visitante = partida['partida']['visitante']

            # Garante que os times existem no acumulador
            if mandante not in team_stats_accumulator:
                team_stats_accumulator[mandante] = {
                    "as_mandante": {"count": 0, "chutes": 0, "chutes_a_gol": 0, "posse_de_bola": 0, "passes": 0, "precisao_de_passe": 0, "faltas": 0, "cartoes_amarelos": 0, "cartoes_vermelhos": 0, "impedimentos": 0, "escanteios": 0, "cruzamentos": 0, "precisao_cruzamento": 0},
                    "as_visitante": {"count": 0, "chutes": 0, "chutes_a_gol": 0, "posse_de_bola": 0, "passes": 0, "precisao_de_passe": 0, "faltas": 0, "cartoes_amarelos": 0, "cartoes_vermelhos": 0, "impedimentos": 0, "escanteios": 0, "cruzamentos": 0, "precisao_cruzamento": 0}
                }
            if visitante not in team_stats_accumulator:
                team_stats_accumulator[visitante] = {
                    "as_mandante": {"count": 0, "chutes": 0, "chutes_a_gol": 0, "posse_de_bola": 0, "passes": 0, "precisao_de_passe": 0, "faltas": 0, "cartoes_amarelos": 0, "cartoes_vermelhos": 0, "impedimentos": 0, "escanteios": 0, "cruzamentos": 0, "precisao_cruzamento": 0},
                    "as_visitante": {"count": 0, "chutes": 0, "chutes_a_gol": 0, "posse_de_bola": 0, "passes": 0, "precisao_de_passe": 0, "faltas": 0, "cartoes_amarelos": 0, "cartoes_vermelhos": 0, "impedimentos": 0, "escanteios": 0, "cruzamentos": 0, "precisao_cruzamento": 0}
                }

            mandante_stats = None
            visitante_stats = None

            for team_name, team_stats in estatisticas.items():
                if team_name == mandante:
                    mandante_stats = team_stats
                elif team_name == visitante:
                    visitante_stats = team_stats

            if not mandante_stats or not visitante_stats:
                continue

            # Acumula estatísticas para o mandante
            team_stats_accumulator[mandante]["as_mandante"]["count"] += 1
            for stat in team_stats_accumulator[mandante]["as_mandante"]:
                if stat != "count":
                    team_stats_accumulator[mandante]["as_mandante"][stat] += get_stat_value(mandante_stats, stat)

            # Acumula estatísticas para o visitante
            team_stats_accumulator[visitante]["as_visitante"]["count"] += 1
            for stat in team_stats_accumulator[visitante]["as_visitante"]:
                if stat != "count":
                    team_stats_accumulator[visitante]["as_visitante"][stat] += get_stat_value(visitante_stats, stat)

    # Calcula as médias
    historical_averages_by_team = {}
    for team, stats in team_stats_accumulator.items():
        team_avg = {}
        for role, role_stats in stats.items():
            count = role_stats["count"]
            if count > 0:
                for stat, total_value in role_stats.items():
                    if stat != "count":
                        team_avg[f"{stat}_media_{role}"] = round(total_value / count, 2)
            else:
                # Caso o time não tenha jogado nessa role, define médias como 0 ou um valor padrão razoável
                for stat in role_stats:
                    if stat != "count":
                        team_avg[f"{stat}_media_{role}"] = 0 # Ou um valor padrão
        historical_averages_by_team[team] = team_avg
    
    return historical_averages_by_team

if __name__ == '__main__':
    medias_por_time = calcular_medias_por_time()
    for team, stats in medias_por_time.items():
        print(f"\nMédias Históricas para {team}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")
