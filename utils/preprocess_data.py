import numpy as np


def preprocess_data(data, n_samples, n_candidates):
  # Create a matrix with binary representation of
  # used candidates for a sample, on each row
  data_matrix = np.zeros((n_samples, n_candidates), dtype=np.byte)

  for index, used_candidate in enumerate(data):
    # Add one only at positions in used_candidates
    data_matrix[index, used_candidate] = 1

  return data_matrix
