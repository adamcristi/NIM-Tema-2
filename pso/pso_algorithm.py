import numpy as np
import copy
import sys
import time
import concurrent.futures

from utils.coverage_check import fast_coverage_check

MAX_WORKERS = 8  # 12


class PSOAlgorithm:

    def __init__(self, data_matrix, eval_function, runs=1, iterations=30, particles=30, inertia_weight=1,
                 acceleration_factor_1=2.05, acceleration_factor_2=2.05):
        # particles = data_matrix
        # best_particle = [particles_like]
        # best_swarm = particle_like

        self.data_matrix = data_matrix
        self.evaluation_function = eval_function
        self.num_runs = runs
        self.num_iterations = iterations
        self.inertia = inertia_weight
        self.acc_fac_1 = acceleration_factor_1
        self.acc_fac_2 = acceleration_factor_2

        self.particles_swarm = None  # (dimensions (30, 2904))
        self.particles_swarm_dimensions = (particles, data_matrix.shape[1])  # (30, 2904)

        self.personal_best_particles_swarm = None  # (dimensions (30, 2904))
        self.global_best_particles_swarm = None  # (dimensions (1, 2904))

        self.velocities_particles_swarm = None  # (dimensions (30, 2904))

        self.min_velocity = -6
        self.max_velocity = 6

    # uniformly distributed
    def initialise_particles(self):
        # init particles
        # compute best_particles
        # compute best_swarm

        self.particles_swarm = []
        counts_ones = np.random.randint(low=0, high=self.particles_swarm_dimensions[1],
                                        size=self.particles_swarm_dimensions[0])

        for count_ones in counts_ones:
            ones_particle = np.ones((count_ones,))
            zeros_particle = np.zeros((self.particles_swarm_dimensions[1] - count_ones,))
            ones_and_zeros_particle = np.concatenate((ones_particle, zeros_particle)).astype(np.int64)
            np.random.shuffle(ones_and_zeros_particle)
            self.particles_swarm.append(ones_and_zeros_particle)

        self.particles_swarm = np.array(self.particles_swarm) 

        # self.particles_swarm = np.random.randint(0, 2, self.particles_swarm_dimensions)

        self.personal_best_particles_swarm = copy.deepcopy(self.particles_swarm)

        self.global_best_particles_swarm = self.particles_swarm[0]
        eval_value_global_best_particles_swarm = self.evaluation_function(particle=self.particles_swarm[0],
                                                                          data_matrix=self.data_matrix)

        for index_particle in range(1, self.particles_swarm_dimensions[0]):
            eval_value_current_particle = self.evaluation_function(particle=self.particles_swarm[index_particle],
                                                                   data_matrix=self.data_matrix)

            if eval_value_current_particle < eval_value_global_best_particles_swarm:
                self.global_best_particles_swarm = self.particles_swarm[index_particle]
                eval_value_global_best_particles_swarm = eval_value_current_particle

    def initialise_velocity(self):
        self.velocities_particles_swarm = np.random.uniform(low=self.min_velocity, high=self.max_velocity,
                                                            size=self.particles_swarm_dimensions)

    def update_particle_velocity(self, index_particle):
        for dimension in range(len(self.particles_swarm[index_particle])):
            rand_1 = np.random.random()
            rand_2 = np.random.random()

            current_velocity = self.velocities_particles_swarm[index_particle][dimension]

            updated_velocity = self.inertia * current_velocity + \
                               self.acc_fac_1 * rand_1 * (
                                       self.personal_best_particles_swarm[index_particle][dimension] -
                                       self.particles_swarm[index_particle][dimension]) + \
                               self.acc_fac_2 * rand_2 * (self.global_best_particles_swarm[dimension] -
                                                          self.particles_swarm[index_particle][dimension])

            self.velocities_particles_swarm[index_particle][dimension] = updated_velocity

    def update_particle_position(self, index_particle):
        for dimension in range(len(self.particles_swarm[index_particle])):
            rand_value = np.random.random()

            sigmoid_value_current_velocity = 1 / (
                    1 + np.exp(-self.velocities_particles_swarm[index_particle][dimension]))

            if rand_value < sigmoid_value_current_velocity:
                self.particles_swarm[index_particle][dimension] = 1
            else:
                self.particles_swarm[index_particle][dimension] = 0

    def execute_threads(self, array, function, *func_args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as th_executor:

            arr_size = len(array)

            workers = 0
            tasks = []

            still_running = True
            element_index = 0

            while still_running:
                full_usage = True

                # If is not the end of population
                if element_index < arr_size:

                    if workers < MAX_WORKERS:
                        full_usage = False
                        workers += 1

                        tasks.append(th_executor.submit(function, element_index, *func_args))

                        element_index += 1

                if full_usage:

                    done, not_done = concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)

                    # Safety mechanism
                    if len(not_done) == 0 and len(done) == 0:
                        still_running = False

                    else:
                        for future in done:
                            # Remove from active tasks
                            tasks.remove(future)

                            # Mark the worker as free
                            workers -= 1

    def process_particle(self, index_particle, personal_best_values):
        self.update_particle_velocity(index_particle)

        self.update_particle_position(index_particle)

        eval_value_current_particle = self.evaluation_function(
            particle=self.particles_swarm[index_particle],
            data_matrix=self.data_matrix)

        if eval_value_current_particle < personal_best_values[index_particle]:
            self.personal_best_particles_swarm[index_particle] = self.particles_swarm[index_particle]
            personal_best_values[index_particle] = eval_value_current_particle

    def execute_algorithm(self):
        for run in range(self.num_runs):

            self.initialise_particles()
            self.initialise_velocity()

            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                start = time.time_ns()
            else:
                start = time.time()

            personal_best_values = []
            for index_particle in range(self.particles_swarm_dimensions[0]):
                personal_best_values.append(self.evaluation_function(
                    particle=self.personal_best_particles_swarm[index_particle],
                    data_matrix=self.data_matrix))

            eval_value_global_best = self.evaluation_function(particle=self.global_best_particles_swarm,
                                                              data_matrix=self.data_matrix)

            for iteration in range(self.num_iterations):

                # for index_particle in range(self.particles_swarm_dimensions[0]):
                #     self.update_particle_velocity(index_particle)
                #
                #     self.update_particle_position(index_particle)
                #
                #     eval_value_current_particle = self.evaluation_function(
                #         particle=self.particles_swarm[index_particle],
                #         data_matrix=self.data_matrix)
                #
                #     if eval_value_current_particle < personal_best_values[index_particle]:
                #         self.personal_best_particles_swarm[index_particle] = self.particles_swarm[index_particle]
                #         personal_best_values[index_particle] = eval_value_current_particle

                self.execute_threads(range(self.particles_swarm_dimensions[0]), self.process_particle,
                                     personal_best_values)

                for index_particle in range(self.particles_swarm_dimensions[0]):
                    if personal_best_values[index_particle] < eval_value_global_best:
                        self.global_best_particles_swarm = self.personal_best_particles_swarm[index_particle]
                        eval_value_global_best = personal_best_values[index_particle]

                if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                    end = time.time_ns()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
                else:
                    end = time.time()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

        print(np.count_nonzero(self.global_best_particles_swarm))
        print(fast_coverage_check(self.global_best_particles_swarm, self.data_matrix))
        print("")
        for index_particle in range(self.particles_swarm_dimensions[0]):
            print(np.count_nonzero(self.personal_best_particles_swarm[index_particle]))
            print(fast_coverage_check(self.personal_best_particles_swarm[index_particle], self.data_matrix))
        # for index_particle in range(self.particles_swarm_dimensions[0]):
        #     print(np.count_nonzero(self.particles_swarm[index_particle]))
