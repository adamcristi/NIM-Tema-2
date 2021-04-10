from utils.coverage_check import detailed_coverage_check
import numpy as np


# The particle not full covered is more important than the chromose full covered with a lot of cameras
def eval_particle_3(particle, data_matrix):
    counts, full_coverage = detailed_coverage_check(particle, data_matrix)

    if not full_coverage:
        return 1 - np.count_nonzero(counts) / data_matrix.shape[0]

    else:
        ratio_used = np.count_nonzero(particle) / data_matrix.shape[1]
        return ratio_used
