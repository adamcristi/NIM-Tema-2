import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
matplotlib.style.use('seaborn-darkgrid')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# eval_particle_1 and inertia assigment soft
#log_name = "eval_particle_1_soft_best_run_log.txt"

# eval_particle_1 and inertia assigment hard
#log_name = "eval_particle_1_hard_best_run_log.txt"

# eval_particle_2 and inertia assigment soft
#log_name = "eval_particle_2_soft_best_run_log.txt"

# eval_particle_2 and inertia assigment hard
log_name = "eval_particle_2_hard_best_run_log.txt"

# test
#log_name = 'test_best_run_log.txt'

# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plots')
PLOTS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plots')


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def create_plot():
    coverages_global_best = []
    are_full_coverages_global_best = []
    evaluation_values_global_best = []

    minimum_coverages_personal_best = []
    are_full_minimum_coverages_personal_best = []
    evaluation_values_minimum_coverages_personal_best = []

    coverages_best_iteration = []
    are_full_coverages_best_iteration = []
    evaluation_values_best_iteration = []

    title_plot = ''
    if 'eval_particle_1' in log_name and 'soft' in log_name:
        title_plot = 'First Evaluation Function and Soft Assignment Inertia Weight'
    elif 'eval_particle_1' in log_name and 'hard' in log_name:
        title_plot = 'First Evaluation Function and Hard Assignment Inertia Weight'
    elif 'eval_particle_2' in log_name and 'soft' in log_name:
        title_plot = 'Second Evaluation Function and Soft Assignment Inertia Weight'
    elif 'eval_particle_2' in log_name and 'hard' in log_name:
        title_plot = 'Second Evaluation Function and Hard Assignment Inertia Weight'

    with open(os.path.join(LOGS_PATH, log_name), "r") as fd:
        current_iteration = fd.readline().strip()

        while current_iteration:
            data_current_iteration = current_iteration.split(" ")
            data_current_iteration = list(filter(filter_data, data_current_iteration))
            data_current_iteration[0] = data_current_iteration[0].replace(':', '')

            current_coverage_global_best = int(data_current_iteration[data_current_iteration.index('coverage_global_best') + 1])
            current_is_full_coverage_global_best = data_current_iteration[data_current_iteration.index('is_full_coverage_global_best') + 1]
            current_evaluation_value_global_best = float(data_current_iteration[data_current_iteration.index('evaluation_value_global_best') + 1])

            coverages_global_best.append(current_coverage_global_best)
            if current_is_full_coverage_global_best == 'True':
                are_full_coverages_global_best.append('Yes')
            else:
                are_full_coverages_global_best.append('No')
            evaluation_values_global_best.append(current_evaluation_value_global_best)

            current_coverage_minimum_coverage_personal_best = int(data_current_iteration[data_current_iteration.index('minimum_coverage_personal_best') + 1])
            current_is_full_minimum_coverage_personal_best = data_current_iteration[data_current_iteration.index('is_full_minimum_coverage_personal_best') + 1]
            current_evaluation_value_minimum_coverage_personal_best = float(data_current_iteration[data_current_iteration.index('evaluation_value_minimum_coverage_personal_best') + 1])

            minimum_coverages_personal_best.append(current_coverage_minimum_coverage_personal_best)
            if current_is_full_minimum_coverage_personal_best == 'True':
                are_full_minimum_coverages_personal_best.append('Yes')
            else:
                are_full_minimum_coverages_personal_best.append('No')
            evaluation_values_minimum_coverages_personal_best.append(current_evaluation_value_minimum_coverage_personal_best)

            current_coverage_best_iteration = int(data_current_iteration[data_current_iteration.index('coverage_best_iteration') + 1])
            current_is_full_coverage_best_iteration = data_current_iteration[data_current_iteration.index('is_full_coverage_best_iteration') + 1]
            current_evaluation_value_best_iteration = float(data_current_iteration[data_current_iteration.index('evaluation_value_best_iteration') + 1])

            coverages_best_iteration.append(current_coverage_best_iteration)
            if current_is_full_coverage_best_iteration == 'True':
                are_full_coverages_best_iteration.append('Yes')
            else:
                are_full_coverages_best_iteration.append('No')
            evaluation_values_best_iteration.append(current_evaluation_value_best_iteration)

            current_iteration = fd.readline().strip()


    data_global = np.array(coverages_global_best).reshape(len(coverages_global_best), 1)
    data_global = np.append(data_global, np.array(evaluation_values_global_best).reshape(len(evaluation_values_global_best), 1), axis=1)
    df_global = pd.DataFrame(data=data_global, columns=["Coverage Global Best", "Evaluation Global Best"])

    data_best = np.array(coverages_best_iteration).reshape(len(coverages_best_iteration), 1)
    data_best = np.append(data_best, np.array(evaluation_values_best_iteration).reshape(len(evaluation_values_best_iteration), 1), axis=1)
    df_best = pd.DataFrame(data=data_best, columns=["Coverage Current Best", "Evaluation Current Best"])

    #data_df = np.array(coverages_global_best).reshape(len(coverages_global_best), 1)
    #data_df = np.append(data_df, np.array(coverages_best_iteration).reshape(len(coverages_best_iteration), 1), axis=1)
    #df = pd.DataFrame(data=data_df, columns=["Coverage Global Best", "Coverage Current Best"])

    data_df = np.array(evaluation_values_global_best).reshape(len(evaluation_values_global_best), 1)
    data_df = np.append(data_df, np.array(evaluation_values_best_iteration).reshape(len(evaluation_values_best_iteration), 1), axis=1)
    df = pd.DataFrame(data=data_df, columns=["Evaluation Global Best", "Evaluation Current Best"])

    figure, axs = plt.subplots(1, 2, figsize=(9.1, 5.1))
    #sns.set_style("darkgrid")
    #print(plt.style.available)
    #plt.style.use("ggplot")
    #plt.figure(facecolor='w')
    #sns.lineplot(data=df)

    #sns.lineplot(x=np.arange(len(coverages_global_best)), y=coverages_global_best, color='blue', ax=axs[0])
    # sns.lineplot(x=np.arange(len(minimum_coverages_personal_best)), y=minimum_coverages_personal_best, color='blue', ax=axs[1])
    #sns.lineplot(x=np.arange(len(coverages_best_iteration)), y=coverages_best_iteration, color='orange',ax=axs[1])

    #sns.lineplot(x=np.arange(len(evaluation_values_global_best)), y=evaluation_values_global_best, color='blue', ax=axs[0])
    # sns.lineplot(x=np.arange(len(evaluation_values_minimum_coverages_personal_best)), y=evaluation_values_minimum_coverages_personal_best, color='blue',ax=axs[1])
    #sns.lineplot(x=np.arange(len(evaluation_values_best_iteration)), y=evaluation_values_best_iteration, color='orange', ax= axs[1])

    #sns.lineplot(data=df_global, ax=axs[0])
    #sns.lineplot(data=df_best, ax=axs[1])

########################################################################################################################
# First plot #

    ax1 = axs[0]
    # sns.lineplot(x=np.arange(len(coverages_global_best)), y=coverages_global_best, color='orange', ax=ax1, linewidth=3.5)
    line_1 = ax1.plot(np.arange(len(coverages_global_best)), coverages_global_best, color='orange', linewidth=3.5, label="Coverage")

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Coverage', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    # ax1.set_xticks(np.arange(0, len(coverages_global_best) + 1, step=len(coverages_global_best) / 5))

    step_yticks_ax1 = (np.max(coverages_global_best) - np.min(coverages_global_best)) / 5
    ax1.set_yticks(np.around(np.arange(np.min(coverages_global_best), np.max(coverages_global_best) + step_yticks_ax1, step=step_yticks_ax1)))

    second_ax1 = ax1.twinx()
    # sns.lineplot(x=np.arange(len(evaluation_values_global_best)), y=evaluation_values_global_best, color='blue', ax=second_ax1)
    line_2 = second_ax1.plot(np.arange(len(evaluation_values_global_best)), evaluation_values_global_best, color='blue', label="Evaluation")

    second_ax1.lines[0].set_linestyle("--")
    second_ax1.set_ylabel('Evaluation', color='blue') #, rotation=270, labelpad=10)
    second_ax1.tick_params(axis='y', labelcolor='blue')
    second_ax1.set_xticks(np.arange(0, len(evaluation_values_global_best) + 1, step=len(evaluation_values_global_best) / 5))

    step_yticks_second_ax1 = (np.max(evaluation_values_global_best) - np.min(evaluation_values_global_best)) / 9
    second_ax1.set_yticks(np.around(np.arange(np.min(evaluation_values_global_best), np.max(evaluation_values_global_best) + step_yticks_second_ax1,
                                              step=step_yticks_second_ax1), decimals=5))

    lines_second_plot = line_1 + line_2
    labels_lines_first_plot = [line.get_label() for line in lines_second_plot]

    second_ax1.legend(lines_second_plot, labels_lines_first_plot, loc=0, frameon=True)
    second_ax1.set_title("Best Global Particle")

########################################################################################################################
# Second plot #

    ax2 = axs[1]
    # sns.lineplot(x=np.arange(len(coverages_best_iteration)), y=coverages_best_iteration, color='orange', ax=ax2, linewidth=3.5, label="Coverage Current Best")
    line_3 = ax2.plot(np.arange(len(coverages_best_iteration)), coverages_best_iteration, color='orange', linewidth=3.5, label="Coverage")

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Coverage', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    # ax2.set_xticks(np.arange(0, len(coverages_best_iteration)+1, step=len(coverages_best_iteration) / 5))

    step_yticks_ax2 = (np.max(coverages_best_iteration) - np.min(coverages_best_iteration)) / 9
    ax2.set_yticks(np.around(np.arange(np.min(coverages_best_iteration), np.max(coverages_best_iteration) + step_yticks_ax2, step=step_yticks_ax2)))

    second_ax2 = ax2.twinx()
    # sns.lineplot(x=np.arange(len(evaluation_values_best_iteration)), y=evaluation_values_best_iteration, color='blue', ax=second_ax2, label="Evaluation Current Best")
    line_4 = second_ax2.plot(np.arange(len(evaluation_values_best_iteration)), evaluation_values_best_iteration, color='blue', label="Evaluation")

    second_ax2.lines[0].set_linestyle("--")
    second_ax2.set_ylabel('Evaluation', color='blue') #, rotation=270, labelpad=10)
    second_ax2.tick_params(axis='y', labelcolor='blue')
    second_ax2.set_xticks(np.arange(0, len(evaluation_values_best_iteration) + 1, step=len(evaluation_values_best_iteration) / 5))

    step_yticks_second_ax2 = (np.max(evaluation_values_best_iteration) - np.min(evaluation_values_best_iteration)) / 15
    second_ax2.set_yticks(np.around(np.arange(np.min(evaluation_values_best_iteration), np.max(evaluation_values_best_iteration) + step_yticks_second_ax2, step=step_yticks_second_ax2), decimals=10))

    lines_second_plot = line_3 + line_4
    labels_lines_second_plot = [line.get_label() for line in lines_second_plot]

    #second_ax2.grid(False, axis='y')
    second_ax2.legend(lines_second_plot, labels_lines_second_plot, loc=0, frameon=True)
    second_ax2.set_title("Best Current Particle")

    #frame_legend_ax2 = legend_ax2.get_frame()
    #frame_legend_ax2.set_facecolor('darkgray')

    #ax2.legend()
    #second_ax2.legend()
    #second_ax2.legend(labels=["Evaluation Current Best"])

    #tick_spacing = 1
    #second_ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title_plot)
    plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))
    #plt.show()


    ##sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5], color='orange')
    #sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5], hue=df.loc[::5, "All Samples Covered"])
    #plt.xlabel("Iterations", fontsize=10)
    #plt.ylabel("Best Chromosome Minimum Candidates", fontsize=10)
    #plt.tick_params(labelsize=9)
    #plt.title(type_eval_chromosome, fontsize=14)
    #plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))


# Create plot
create_plot()
