import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'logs', 'final')

#eval_type = 'eval1'
eval_type = 'eval2'

#update_type = 'mutation_update'
update_type = 'standard_update'


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def analyse_global_best():

    for root, dirs, files in os.walk(os.path.join(LOGS_PATH, update_type, eval_type)):
        for log_name in files:
            if 'iterations' in log_name:
                print(log_name)
                log_path = os.path.join(root, log_name)

                with open(log_path.rsplit("_", 1)[0] + "_parameters.txt", "r") as fd:
                    data_file = fd.read(4096)
                    if 'eval_particle_1' in data_file:
                        print('eval_particle=eval_particle_1')
                    elif 'eval_particle_2' in data_file:
                        print('eval_particle=eval_particle_2')

                    if 'inertia_assignation_type=soft' in data_file:
                        print('inertia_assignation_type=soft')
                    elif 'inertia_assignation_type=hard' in data_file:
                        print('inertia_assignation_type=hard')

                with open(log_path, "r") as fd:

                    first_coverages_global_best = []
                    last_coverages_global_best = []

                    current_iteration = fd.readline().strip()

                    while current_iteration:
                        if 'Run' in current_iteration:
                            num_current_run = int(current_iteration.split(" ")[1].replace("\n", ""))
                        else:
                            data_current_iteration = current_iteration.split(" ")
                            data_current_iteration = list(filter(filter_data, data_current_iteration))
                            data_current_iteration[0] = data_current_iteration[0].replace(':', '')

                            current_coverage_global_best = int(data_current_iteration[data_current_iteration.index('coverage_global_best') + 1])
                            current_is_full_coverage_global_best = data_current_iteration[data_current_iteration.index('is_full_coverage_global_best') + 1]

                            if current_is_full_coverage_global_best == 'True':
                                if len(first_coverages_global_best) < num_current_run + 1:
                                    first_coverages_global_best.append(current_coverage_global_best)
                                    last_coverages_global_best.append(current_coverage_global_best)
                                elif current_coverage_global_best < last_coverages_global_best[num_current_run]:
                                    last_coverages_global_best[num_current_run] = current_coverage_global_best

                        current_iteration = fd.readline()
                        if current_iteration == '\n':
                            current_iteration = fd.readline().strip()
                        else:
                            current_iteration = current_iteration.strip()

                    first_coverages_global_best = np.array(first_coverages_global_best)
                    last_coverages_global_best = np.array(last_coverages_global_best)
                    difference_coverages_global_best = first_coverages_global_best - last_coverages_global_best
                    # print(first_coverages_global_best)
                    # print(last_coverages_global_best)
                    # print(f"Min: {np.min(last_coverages_global_best)}")
                    # print(f"Max: {np.max(last_coverages_global_best)}")
                    # print(f"Mean: {np.mean(last_coverages_global_best)}")
                    # print(f"Std: {np.std(last_coverages_global_best)}")
                    # print()
                    print(difference_coverages_global_best)
                    print(f"Min: {np.min(difference_coverages_global_best)}")
                    print(f"Max: {np.max(difference_coverages_global_best)}")
                    print(f"Mean: {np.mean(difference_coverages_global_best)}")
                    print(f"Std: {np.std(difference_coverages_global_best)}")
                    print()


# Analyse global best
analyse_global_best()
