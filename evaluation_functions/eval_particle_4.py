from utils.coverage_check import detailed_coverage_check
import numpy as np


def eval_particle_4(particle, data_matrix):
    counts, full_coverage = detailed_coverage_check(particle, data_matrix)

    alpha = 0.1

    return (alpha * np.count_nonzero(particle) / data_matrix.shape[1] +
            (1 - alpha) * (1 - np.count_nonzero(counts) / data_matrix.shape[0]))

    # if not full_coverage:
    #     return 2 - np.count_nonzero(counts) / data_matrix.shape[0]
    #
    # else:
    #     ratio_used = np.count_nonzero(particle) / data_matrix.shape[1]
    #     return ratio_used
