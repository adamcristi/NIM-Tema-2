import numpy as np


def fast_coverage_check(particle, data_matrix):
    """
      This function checks only if there is full coverage.
    """
    # Apply & between the chromosome and each row of data_matrix.
    # If there is ANY value of 1 ("at least one camera covering that sample")
    # in the result (applied by row), then the corresponding sample is covered.
    # Having the coverage for each sample, check if ALL samples are covered.
    return np.all(np.any(data_matrix & particle, axis=1))


def detailed_coverage_check(particle, data_matrix):
    """
      This function checks if there is full coverage and provides the coverage counts (the number of candidates used per sample).
    """
    # Apply & between the chromosome and each row of data_matrix.
    # Count the 1 values in the result (applied by row).
    counts = np.count_nonzero(data_matrix & particle, axis=1)
    # Return the counts and True if there is full coverage
    return counts, np.all(counts)
