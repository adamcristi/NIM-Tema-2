import numpy as np
import copy
import sys
import time
import concurrent.futures
from datetime import datetime
from utils.coverage_check import fast_coverage_check
from path import LOGS_PATH, name

# MAX_WORKERS = 12


class PSOAlgorithm:

    def __init__(self, data_matrix, eval_function, runs=1, iterations=30, particles=30, inertia_weight=1,
                 acceleration_factor_1=2.05, acceleration_factor_2=2.05,
                 experiment_type="_experiment_pso_", inertia_type="soft"):  # experiment_type="_experiment_pso_2_"
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
        self.best_iteration = None

        self.velocities_particles_swarm = None  # (dimensions (30, 2904))
        self.min_velocity = -6
        self.max_velocity = 6

        self.eval_value_global_best_particles = None  # (dimension 1))
        self.eval_values_personal_best_particles = []  # (dimension (30,))
        self.eval_value_best_iteration = None

        self.experiment_name = name + experiment_type + str(datetime.timestamp(datetime.now())) + ".txt"

        self.coverage_global_best = None
        self.is_full_coverage_global_best = None
        self.evaluation_value_global_best = None
        self.minimum_coverage_personal_best = None
        self.is_full_minimum_coverage_personal_best = None
        self.evaluation_value_minimum_coverage_personal_best = None
        self.coverage_best_iteration = None
        self.is_full_coverage_best_iteration = None
        self.evaluation_value_best_iteration = None

        self.runs_coverage_global_best = []
        self.runs_is_full_coverage_global_best = []
        self.runs_evaluation_value_global_best = []
        self.runs_minimum_coverage_personal_best = []
        self.runs_is_full_minimum_coverage_personal_best = []
        self.runs_evaluation_value_minimum_coverage_personal_best = []
        self.runs_coverage_best_iteration = []
        self.runs_is_full_coverage_best_iteration = []
        self.runs_evaluation_value_best_iteration = []

        self.inertia_max = 0.6
        self.inertia_min = 0.2
        self.inertia_diff = self.inertia_max - self.inertia_min

        self.species_count = 5
        self.species_indecies = np.cumsum([particles / self.species_count for _ in range(self.species_count)]).astype(np.int)

        self.iws_functions = [self.iws_1, self.iws_2, self.iws_3, self.iws_4, self.iws_3]

        self.current_iteration = 0

        self.inertia_type = inertia_type

    ########################################################################################################################
    # Initialization #

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

        # Old initialization
        # self.particles_swarm = np.random.randint(0, 2, self.particles_swarm_dimensions)

        self.personal_best_particles_swarm = copy.deepcopy(self.particles_swarm)

        self.global_best_particles_swarm = copy.deepcopy(self.particles_swarm[0])
        self.eval_values_personal_best_particles = self.evaluation_function(particle=self.particles_swarm[0],
                                                                            data_matrix=self.data_matrix)

        for index_particle in range(1, self.particles_swarm_dimensions[0]):
            eval_value_current_particle = self.evaluation_function(particle=self.particles_swarm[index_particle],
                                                                   data_matrix=self.data_matrix)

            if eval_value_current_particle < self.eval_values_personal_best_particles:
                self.global_best_particles_swarm = copy.deepcopy(self.particles_swarm[index_particle])
                self.eval_values_personal_best_particles = eval_value_current_particle

    def initialise_velocity(self):
        self.velocities_particles_swarm = np.random.uniform(low=self.min_velocity, high=self.max_velocity,
                                                            size=self.particles_swarm_dimensions)

    ########################################################################################################################
    # Inertia schemes #

    def iws_1(self):
        return self.inertia_max - self.inertia_diff * (self.current_iteration / self.num_iterations)
        # return self.inertia_min + self.inertia_diff * (self.current_iteration / self.num_iterations)

    def iws_2(self):
        return 0.5 * (1 + np.random.rand())

    def iws_3(self):
        return ((self.num_iterations - self.current_iteration) ** 1.2 / self.num_iterations ** 1.2) * \
               self.inertia_diff + self.inertia_min

    def iws_4(self):
        return self.inertia

    ########################################################################################################################
    # Updating #

    def update_particle_velocity(self, index_particle, inertia_function=None):
        for dimension in range(len(self.particles_swarm[index_particle])):
            rand_1 = np.random.random()
            rand_2 = np.random.random()

            current_velocity = self.velocities_particles_swarm[index_particle][dimension]

            if inertia_function is None:
                rand_3 = np.random.randint(0, 4)
                inertia_function = self.iws_functions[rand_3]

            updated_velocity = inertia_function() * current_velocity + \
                               self.acc_fac_1 * rand_1 * (
                                       self.personal_best_particles_swarm[index_particle][dimension] -
                                       self.particles_swarm[index_particle][dimension]) + \
                               self.acc_fac_2 * rand_2 * (self.global_best_particles_swarm[dimension] -
                                                          self.particles_swarm[index_particle][dimension])

            self.velocities_particles_swarm[index_particle][dimension] = updated_velocity

    # def update_global_best_particle_velocity(self, index_particle, inertia_function):
    #     for dimension in range(len(self.particles_swarm[index_particle])):
    #
    #         current_velocity = self.velocities_particles_swarm[index_particle][dimension]
    #
    #         updated_velocity = inertia_function() * current_velocity \
    #                            - self.particles_swarm[index_particle][dimension] \
    #                            + self.global_best_particles_swarm[dimension] \
    #                            + self.ro * (1 - 2 * np.random.rand())
    #
    #         # rand_1 = np.random.random()
    #         # rand_2 = np.random.random()
    #         #
    #         # current_velocity = self.velocities_particles_swarm[index_particle][dimension]
    #         #
    #         # updated_velocity = inertia_function() * current_velocity + \
    #         #                    self.acc_fac_1 * rand_1 * (
    #         #                            self.personal_best_particles_swarm[index_particle][dimension] -
    #         #                            self.particles_swarm[index_particle][dimension]) + \
    #         #                    self.acc_fac_2 * rand_2 * (self.global_best_particles_swarm[dimension] -
    #         #                                               self.particles_swarm[index_particle][dimension])
    #
    #         self.velocities_particles_swarm[index_particle][dimension] = updated_velocity

    def update_particle_position(self, index_particle):
        for dimension in range(len(self.particles_swarm[index_particle])):
            rand_value = np.random.random()

            sigmoid_value_current_velocity = 2 * np.abs(1 / (
                    1 + np.exp(-self.velocities_particles_swarm[index_particle][dimension])) - 0.5)

            if rand_value < sigmoid_value_current_velocity:
                self.particles_swarm[index_particle][dimension] = 1 - self.particles_swarm[index_particle][dimension]
            else:
                self.particles_swarm[index_particle][dimension] = self.particles_swarm[index_particle][dimension]

    ########################################################################################################################
    # Multithreading #

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

    def process_particle(self, index_particle):

        # function_index = 0
        # for index in self.species_indecies:
        #     if index > index_particle:
        #         function_index = index
        #         break

        if np.all(self.particles_swarm[index_particle] == self.global_best_particles_swarm):
            print("SAME HERE")

        self.update_particle_velocity(index_particle)

        self.update_particle_position(index_particle)

        eval_value_current_particle = self.evaluation_function(particle=self.particles_swarm[index_particle],
                                                               data_matrix=self.data_matrix)

        if eval_value_current_particle < self.eval_values_personal_best_particles[index_particle]:
            self.personal_best_particles_swarm[index_particle] = self.particles_swarm[index_particle]
            self.eval_values_personal_best_particles[index_particle] = eval_value_current_particle

    # def process_particle(self, index_particle, personal_best_values):
    #    self.update_particle_velocity(index_particle)
    #
    #    self.update_particle_position(index_particle)
    #
    #    eval_value_current_particle = self.evaluation_function(
    #        particle=self.particles_swarm[index_particle],
    #        data_matrix=self.data_matrix)
    #
    #    if eval_value_current_particle < personal_best_values[index_particle]:
    #        self.personal_best_particles_swarm[index_particle] = self.particles_swarm[index_particle]
    #        personal_best_values[index_particle] = eval_value_current_particle

    ########################################################################################################################
    # Logging #

    def write_log_parameters(self):
        with open(LOGS_PATH + self.experiment_name[:-4] + "_parameters.txt", "w") as file:
            parameters = f"{self.experiment_name[:-4]}\n" \
                         + f"number_particles={self.particles_swarm_dimensions[0]}\n" \
                         + f"runs={self.num_runs}\n" \
                         + f"iterations={self.num_iterations}\n" \
                         + f"eval_particle={self.evaluation_function.__name__}\n" \
                         + f"inertia_weight={self.inertia}\n" \
                         + f"acceleration_factor_1={self.acc_fac_1}\n" \
                         + f"acceleration_factor_2={self.acc_fac_2}\n" \
                         + f"minimum_velocity={self.min_velocity}\n" \
                         + f"maximum_velocity={self.max_velocity}\n" \
                         + f"inertia_assignation_type={self.inertia_type}\n"

            file.write(parameters)

    def write_number_run(self, number_run):
        with open(LOGS_PATH + self.experiment_name[:-4] + "_iterations.txt", "a+") as file:
            if number_run == 0:
                file.write(f"Run {number_run} \n\n")
            else:
                file.write(f"\nRun {number_run} \n\n")

    def get_log_info(self):
        self.coverage_global_best = np.count_nonzero(self.global_best_particles_swarm)
        self.is_full_coverage_global_best = fast_coverage_check(self.global_best_particles_swarm, self.data_matrix)
        self.evaluation_value_global_best = self.eval_value_global_best_particles

        coverages_personal_best_particles_swarm = [np.count_nonzero(self.personal_best_particles_swarm[index_particle])
                                                   for index_particle in range(self.particles_swarm_dimensions[0])]

        self.minimum_coverage_personal_best = np.min(coverages_personal_best_particles_swarm)
        index_minimum_coverage_personal_best = np.argmin(coverages_personal_best_particles_swarm)
        self.is_full_minimum_coverage_personal_best = fast_coverage_check(
            self.personal_best_particles_swarm[index_minimum_coverage_personal_best],
            self.data_matrix)
        self.evaluation_value_minimum_coverage_personal_best = self.eval_values_personal_best_particles[
            index_minimum_coverage_personal_best]

        self.coverage_best_iteration = np.count_nonzero(self.best_iteration)
        self.is_full_coverage_best_iteration = fast_coverage_check(self.best_iteration, self.data_matrix)
        self.evaluation_value_best_iteration = self.eval_value_best_iteration

        delimiter = " ;" + " " * 4

        info = f"coverage_global_best = {self.coverage_global_best}{delimiter}"
        info += f"is_full_coverage_global_best = {self.is_full_coverage_global_best}{delimiter}"
        info += f"evaluation_value_global_best = {self.evaluation_value_global_best:.14f}{delimiter}"
        info += f"minimum_coverage_personal_best = {self.minimum_coverage_personal_best}{delimiter}"
        info += f"is_full_minimum_coverage_personal_best = {self.is_full_minimum_coverage_personal_best}{delimiter}"
        info += f"evaluation_value_minimum_coverage_personal_best = {self.evaluation_value_minimum_coverage_personal_best:.14f}{delimiter}"
        info += f"coverage_best_iteration = {self.coverage_best_iteration}{delimiter}"
        info += f"is_full_coverage_best_iteration = {self.is_full_coverage_best_iteration}{delimiter}"
        info += f"evaluation_value_best_iteration = {self.evaluation_value_best_iteration}{delimiter}"

        return info

    def write_log_info_iteration(self, number_iteration):
        info_iteration = f"Iteration {number_iteration}: "
        info_iteration += self.get_log_info()

        with open(LOGS_PATH + self.experiment_name[:-4] + "_iterations.txt", "a+") as file:
            file.write(info_iteration + "\n")

    def write_log_info_run(self, number_run):
        info_run = f"Run {number_run}: "
        info_run += self.get_log_info()

        with open(LOGS_PATH + self.experiment_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_run + "\n")

        if self.is_full_coverage_global_best:
            self.runs_coverage_global_best.append(self.coverage_global_best)
            self.runs_is_full_coverage_global_best.append(self.is_full_coverage_global_best)
            self.runs_evaluation_value_global_best.append(self.evaluation_value_global_best)
            self.runs_minimum_coverage_personal_best.append(self.minimum_coverage_personal_best)
            self.runs_is_full_minimum_coverage_personal_best.append(self.is_full_minimum_coverage_personal_best)
            self.runs_evaluation_value_minimum_coverage_personal_best.append(self.evaluation_value_minimum_coverage_personal_best)
            self.runs_coverage_best_iteration.append(self.coverage_best_iteration)
            self.runs_is_full_coverage_best_iteration.append(self.is_full_coverage_best_iteration)
            self.runs_evaluation_value_best_iteration.append(self.evaluation_value_best_iteration)

    def write_log_info_runs(self):
        info_runs = f"\n\nRuns: "

        index_min_run_coverage_global_best = np.argmin(self.runs_coverage_global_best)
        index_min_run_minimum_coverage_personal_best = np.argmin(self.runs_minimum_coverage_personal_best)
        index_min_run_coverage_best_iteration = np.argmin(self.runs_coverage_best_iteration)

        delimiter = " ;" + " " * 4

        info_runs += f"is_full_min_coverage_global_best = {self.runs_is_full_coverage_global_best[index_min_run_coverage_global_best]}{delimiter}"
        info_runs += f"min_coverage_global_best = {self.runs_coverage_global_best[index_min_run_coverage_global_best]}{delimiter}"
        info_runs += f"max_coverage_global_best = {np.max(self.runs_coverage_global_best)}{delimiter}"
        info_runs += f"mean_coverage_global_best = {np.mean(self.runs_coverage_global_best)}{delimiter}"
        info_runs += f"std_coverage_global_best = {np.std(self.runs_coverage_global_best)}{delimiter}"

        info_runs += f"\n      "
        info_runs += f"is_full_min_run_minimum_coverage_personal_best = {self.runs_is_full_minimum_coverage_personal_best[index_min_run_minimum_coverage_personal_best]}{delimiter}"
        info_runs += f"min_minimum_coverage_personal_best = {self.runs_minimum_coverage_personal_best[index_min_run_minimum_coverage_personal_best]}{delimiter}"
        info_runs += f"max_minimum_coverage_personal_best = {np.max(self.runs_minimum_coverage_personal_best)}{delimiter}"
        info_runs += f"mean_minimum_coverage_personal_best = {np.mean(self.runs_minimum_coverage_personal_best)}{delimiter}"
        info_runs += f"std_minimum_coverage_personal_best = {np.std(self.runs_minimum_coverage_personal_best)}{delimiter}"

        info_runs += f"\n      "
        info_runs += f"is_full_coverage_best_iteration = {self.runs_is_full_coverage_best_iteration[index_min_run_coverage_best_iteration]}{delimiter}"
        info_runs += f"min_coverage_best_iteration = {self.runs_coverage_best_iteration[index_min_run_coverage_best_iteration]}{delimiter}"
        info_runs += f"max_coverage_best_iteration = {np.max(self.runs_coverage_best_iteration)}{delimiter}"
        info_runs += f"mean_coverage_best_iteration = {np.mean(self.runs_coverage_best_iteration)}{delimiter}"
        info_runs += f"std_coverage_best_iteration = {np.std(self.runs_coverage_best_iteration)}{delimiter}"

        with open(LOGS_PATH + self.experiment_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_runs + "\n")

    ########################################################################################################################
    # Execution #

    def execute_algorithm(self):
        self.write_log_parameters()

        for run in range(self.num_runs):
            if run == 0:
                print(f"Run {run}")
            else:
                print(f"\nRun {run}")

            self.current_iteration = 0

            self.write_number_run(run)

            self.initialise_particles()
            self.initialise_velocity()

            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                start = time.time_ns()
            else:
                start = time.time()

            self.eval_values_personal_best_particles = []
            for index_particle in range(self.particles_swarm_dimensions[0]):
                self.eval_values_personal_best_particles.append(
                    self.evaluation_function(particle=self.personal_best_particles_swarm[index_particle],
                                             data_matrix=self.data_matrix))

            self.eval_value_global_best_particles = self.evaluation_function(particle=self.global_best_particles_swarm,
                                                                             data_matrix=self.data_matrix)

            # coverages_personal_best_particles_swarm = [
            #     np.count_nonzero(self.personal_best_particles_swarm[index_particle])
            #     for index_particle in range(self.particles_swarm_dimensions[0])]
            #
            # print(np.count_nonzero(self.global_best_particles_swarm))
            # print(self.eval_value_global_best_particles)
            # print(coverages_personal_best_particles_swarm)
            # print(self.eval_values_personal_best_particles)

            for iteration in range(self.num_iterations):

                self.current_iteration = iteration

                self.best_iteration= copy.deepcopy(self.particles_swarm[0])
                self.eval_value_best_iteration = self.evaluation_function(particle=self.best_iteration, data_matrix=self.data_matrix)

                for index_particle in range(self.particles_swarm_dimensions[0]):

                    if self.inertia_type == "soft":
                        self.update_particle_velocity(index_particle)

                    else:
                        function_index = 0
                        for index, boundary in enumerate(self.species_indecies):
                            if index_particle < boundary:
                                function_index = index
                                break

                        self.update_particle_velocity(index_particle, self.iws_functions[function_index])

                    self.update_particle_position(index_particle)

                    eval_value_current_particle = self.evaluation_function(
                        particle=self.particles_swarm[index_particle],
                        data_matrix=self.data_matrix)

                    if eval_value_current_particle < self.eval_values_personal_best_particles[index_particle]:
                        self.personal_best_particles_swarm[index_particle] = copy.deepcopy(
                            self.particles_swarm[index_particle])
                        self.eval_values_personal_best_particles[index_particle] = eval_value_current_particle

                        if self.eval_value_best_iteration > eval_value_current_particle:
                            self.eval_value_best_iteration = eval_value_current_particle
                            self.best_iteration = copy.deepcopy(self.particles_swarm[index_particle])

                # self.execute_threads(range(self.particles_swarm_dimensions[0]), self.process_particle)
                # self.eval_values_personal_best_particles)

                for index_particle in range(self.particles_swarm_dimensions[0]):
                    if self.eval_values_personal_best_particles[index_particle] < self.eval_value_global_best_particles:
                        self.global_best_particles_swarm = copy.deepcopy(
                            self.personal_best_particles_swarm[index_particle])
                        self.eval_value_global_best_particles = self.eval_values_personal_best_particles[index_particle]

                    # current_eval = self.evaluation_function(particle=self.particles_swarm[index_particle], data_matrix=self.data_matrix)
                    # if self.eval_value_best_iteration > current_eval:
                    #     self.eval_value_best_iteration = current_eval
                    #     self.best_iteration= copy.deepcopy(self.particles_swarm[index_particle])

                # print(self.eval_value_best_iteration)
                # print(self.velocities_particles_swarm)

                if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                    end = time.time_ns()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
                else:
                    end = time.time()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

                # coverages_personal_best_particles_swarm = [
                #     np.count_nonzero(self.personal_best_particles_swarm[index_particle])
                #     for index_particle in range(self.particles_swarm_dimensions[0])]
                #
                # print(np.count_nonzero(self.global_best_particles_swarm))
                # print(self.eval_value_global_best_particles)
                # print(coverages_personal_best_particles_swarm)
                # print(self.eval_values_personal_best_particles)

                self.write_log_info_iteration(iteration)

            self.write_log_info_run(run)

        self.write_log_info_runs()

        # print(np.count_nonzero(self.global_best_particles_swarm))
        # print(fast_coverage_check(self.global_best_particles_swarm, self.data_matrix))
        # print("")
        # for index_particle in range(self.particles_swarm_dimensions[0]):
        #     print(np.count_nonzero(self.personal_best_particles_swarm[index_particle]))
        #     print(fast_coverage_check(self.personal_best_particles_swarm[index_particle], self.data_matrix))
        # # for index_particle in range(self.particles_swarm_dimensions[0]):
        # #     print(np.count_nonzero(self.particles_swarm[index_particle]))
