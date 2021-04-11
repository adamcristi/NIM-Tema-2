from utils.coverage_check import fast_coverage_check
import numpy as np


def eval_particle_2(particle, data_matrix):
    full_coverage = fast_coverage_check(particle, data_matrix)

    ratio = np.count_nonzero(particle) / data_matrix.shape[1]

    return ratio if full_coverage else 1 - ratio
