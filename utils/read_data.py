
def read_data(path):
    data = []
    # The sum of all camera counts per sample
    total_used_sum = 0

    with open(path, "r") as file:
        n_samples, n_candidates = tuple(map(int, file.readline().split()))

        current_sample = file.readline().strip()
        while current_sample:
            current_candidates_count = int(file.readline())
            total_used_sum += current_candidates_count

            data += [[int(x) for x in file.readline().split()]]

            current_sample = file.readline().strip()

    return data, n_samples, n_candidates, total_used_sum
