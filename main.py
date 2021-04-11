import sys
import time

from evaluation_functions.eval_particle_4 import eval_particle_4
from utils.preprocess_data import preprocess_data
from utils.read_data import read_data
from evaluation_functions.eval_particle_1 import eval_particle_1
from evaluation_functions.eval_particle_2 import eval_particle_2
from evaluation_functions.eval_particle_3 import eval_particle_3
from pso.pso_algorithm import PSOAlgorithm

from path import LOGS_PATH, path, name

data, n_samples, n_candidates, total_used_sum = read_data(path)

data_matrix = preprocess_data(data, n_samples, n_candidates)

# Particle Swarm Optimization Algorithm

if sys.version_info.major == 3 and sys.version_info.minor >= 7:
    start = time.time_ns()
else:
    start = time.time()

pso = PSOAlgorithm(data_matrix=data_matrix,
                   eval_function=eval_particle_4,
                   runs=2,
                   iterations=100,
                   particles=50,
                   inertia_weight=0.2,
                   acceleration_factor_1=2.05,
                   acceleration_factor_2=2.05,
                   experiment_type="_experiment_pso_")

pso.execute_algorithm()

if sys.version_info.major == 3 and sys.version_info.minor >= 7:
    end = time.time_ns()
    print(f"Total time: {(end-start) / 1e9} seconds.")
else:
    end = time.time()
    print(f"Total time: {(end - start)} seconds.")


