from math import ceil, floor

from pypet import load_trajectory, pypetconstants
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

# from save_legend import save_legend


def plot_fee_losses_over_time(filename):

    outputs_directory = str(Path("../outputs").resolve())
    save_at_directory = outputs_directory + "/figures/"
    Path(save_at_directory).mkdir(parents=True, exist_ok=True)

    times = 'total_fortune_including_pending_swaps_times'
    # result = 'total_fortune_including_pending_swaps_values'
    fee_losses_over_time = 'fee_losses_over_time'
    rebalancing_fees_over_time = 'rebalancing_fees_over_time'

    number_of_points_to_plot = 200

    traj = load_trajectory(filename=outputs_directory + '/results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_all=pypetconstants.LOAD_DATA)
    # traj = load_trajectory(filename=outputs_directory + '/results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_parameters=2, load_results=1)
    # traj.v_auto_load = True

    # Parse parameter values

    par_rebalancing_policy_values = traj.f_get('rebalancing_policy').f_get_range()
    par_rebalancing_policy_values = list(dict.fromkeys(par_rebalancing_policy_values))

    # Parse results

    result_times = list(traj.f_get_from_runs(times, fast_access=True).values())
    # result_times = np.reshape(np.array(result_times), (len(par_rebalancing_policy_values), len(result_times)/len(par_rebalancing_policy_values)))
    fee_losses_over_time_values = list(traj.f_get_from_runs(fee_losses_over_time, fast_access=True).values())
    # result_values = np.reshape(np.array(result_values), (len(par_rebalancing_policy_values), len(result_times)/len(par_rebalancing_policy_values)))
    rebalancing_fees_over_time_values = list(traj.f_get_from_runs(rebalancing_fees_over_time, fast_access=True).values())

    # Process fees

    cumulative_fee_losses_over_time_values = np.cumsum(fee_losses_over_time_values, axis=1)     # sum over rows
    cumulative_rebalancing_fees_over_time_values = np.cumsum(rebalancing_fees_over_time_values, axis=1)     # sum over rows
    cumulative_total_cost_values = cumulative_fee_losses_over_time_values + cumulative_rebalancing_fees_over_time_values

    # Plot results

    number_of_total_time_points = len(result_times[0])
    number_of_points_to_plot = min(number_of_points_to_plot, number_of_total_time_points)
    step_size = floor(number_of_total_time_points / number_of_points_to_plot)
    result_times_to_plot = {}
    cumulative_fee_losses_over_time_values_to_plot = {}
    cumulative_rebalancing_fees_over_time_values_to_plot = {}
    cumulative_total_cost_values_to_plot = {}
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        result_times_to_plot[rebalancing_policy] = result_times[rebalancing_policy_index][::step_size]
        if len(result_times[rebalancing_policy_index]) % step_size != 0:
            result_times_to_plot[rebalancing_policy].append(result_times[rebalancing_policy_index][-1])
        cumulative_fee_losses_over_time_values_to_plot[rebalancing_policy] = list(cumulative_fee_losses_over_time_values[rebalancing_policy_index][::step_size])
        if len(cumulative_fee_losses_over_time_values[rebalancing_policy_index]) % step_size != 0:
            cumulative_fee_losses_over_time_values_to_plot[rebalancing_policy].append(cumulative_fee_losses_over_time_values[rebalancing_policy_index][-1])
        cumulative_rebalancing_fees_over_time_values_to_plot[rebalancing_policy] = list(cumulative_rebalancing_fees_over_time_values[rebalancing_policy_index][::step_size])
        if len(cumulative_rebalancing_fees_over_time_values[rebalancing_policy_index]) % step_size != 0:
            cumulative_rebalancing_fees_over_time_values_to_plot[rebalancing_policy].append(cumulative_rebalancing_fees_over_time_values[rebalancing_policy_index][-1])
        cumulative_total_cost_values_to_plot[rebalancing_policy] = list(cumulative_total_cost_values[rebalancing_policy_index][::step_size])
        if len(cumulative_total_cost_values[rebalancing_policy_index]) % step_size != 0:
            cumulative_total_cost_values_to_plot[rebalancing_policy].append(cumulative_total_cost_values[rebalancing_policy_index][-1])

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrkmcy')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({"errorbar.capsize": 3})
    markers = [".", "x", "s", "+", "*", "d"]


    fig, ax1 = plt.subplots()
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        innermost_index = rebalancing_policy_index

        ax1.plot(result_times_to_plot[rebalancing_policy], cumulative_total_cost_values_to_plot[rebalancing_policy], label=rebalancing_policy, linestyle=linestyles[0], color=colors[innermost_index], alpha=1)

        ax1.grid(True)
        # ax1.set_ylim(ymin=0)
        ax1.set_xlabel("Time (minutes)")
        ax1.set_ylabel("Total cost (tx + reb. fees) ($)")

        lines, labels = ax1.get_legend_handles_labels()
        legend = ax1.legend(lines, labels, loc='best')

        fig.savefig(save_at_directory + filename + "_total_losses_over_time.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_fees_over_time.pdf", bbox_inches='tight')

        # legend_filename = filename + "_legend.png"
        # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

    fig, ax2 = plt.subplots()
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        innermost_index = rebalancing_policy_index

        ax2.plot(result_times_to_plot[rebalancing_policy], cumulative_fee_losses_over_time_values_to_plot[rebalancing_policy], label=rebalancing_policy, linestyle=linestyles[0], color=colors[innermost_index], alpha=1)

        ax2.grid(True)
        # ax1.set_ylim(ymin=0)
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Total tx fee losses ($)")

        lines, labels = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines, labels, loc='best')

        fig.savefig(save_at_directory + filename + "_tx_fee_losses_over_time.png", bbox_inches='tight')
        fig.savefig(save_at_directory + filename + "_tx_fee_losses_over_time.pdf", bbox_inches='tight')

        # legend_filename = filename + "_legend.png"
        # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

    fig, ax3 = plt.subplots()
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        innermost_index = rebalancing_policy_index

        ax3.plot(result_times_to_plot[rebalancing_policy], cumulative_rebalancing_fees_over_time_values_to_plot[rebalancing_policy], label=rebalancing_policy, linestyle=linestyles[0], color=colors[innermost_index], alpha=1)

        ax3.grid(True)
        # ax1.set_ylim(ymin=0)
        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Total rebalancing fees ($)")

        lines, labels = ax3.get_legend_handles_labels()
        legend = ax3.legend(lines, labels, loc='best')

        fig.savefig(save_at_directory + filename + "_reb_fees_over_time.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_fees_over_time.pdf", bbox_inches='tight')

        # legend_filename = filename + "_legend.png"
        # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

    plt.show()


if __name__ == '__main__':
    plot_fee_losses_over_time(filename='results_05')
