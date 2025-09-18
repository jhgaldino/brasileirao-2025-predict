import os

# Caminhos padrao
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset.json')
CLASSIFICATION_PATH = os.path.join(PROJECT_ROOT, 'classificacao.json')
NEXT_ROUND_PATH = os.path.join(PROJECT_ROOT, 'next_round.json')

# Parametros de modelo (defaults gerais)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Mapeamentos de classes
CLASS_MAP = {'H': 0, 'D': 1, 'A': 2}
CLASS_MAP_INV = {0: 'H', 1: 'D', 2: 'A'}


