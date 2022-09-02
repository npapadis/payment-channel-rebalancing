from math import ceil, floor

from pypet import load_trajectory, pypetconstants
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

# from save_legend import save_legend


def plot_total_fortune_over_time(filename):

    outputs_directory = str(Path("../outputs").resolve())
    save_at_directory = outputs_directory + "/figures/"
    Path(save_at_directory).mkdir(parents=True, exist_ok=True)

    times = 'total_fortune_including_pending_swaps_times'
    result = 'total_fortune_including_pending_swaps_values'
    # result = 'total_fortune_including_pending_swaps_minus_losses_values'

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
    result_values = list(traj.f_get_from_runs(result, fast_access=True).values())
    # result_values = np.reshape(np.array(result_values), (len(par_rebalancing_policy_values), len(result_times)/len(par_rebalancing_policy_values)))

    number_of_total_time_points = len(result_times[0])
    number_of_points_to_plot = min(number_of_points_to_plot, number_of_total_time_points)
    step_size = floor(number_of_total_time_points / number_of_points_to_plot)
    result_times_to_plot = {}
    result_values_to_plot = {}
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        result_times_to_plot[rebalancing_policy] = result_times[rebalancing_policy_index][::step_size]
        if len(result_times[rebalancing_policy_index]) % step_size != 0:
            result_times_to_plot[rebalancing_policy].append(result_times[rebalancing_policy_index][-1])
        result_values_to_plot[rebalancing_policy] = result_values[rebalancing_policy_index][::step_size]
        if len(result_values[rebalancing_policy_index]) % step_size != 0:
            result_values_to_plot[rebalancing_policy].append(result_values[rebalancing_policy_index][-1])

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrkmcy')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({"errorbar.capsize": 3})
    markers = [".", "x", "s", "+", "*", "d"]


    fig, ax1 = plt.subplots()
    for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
        innermost_index = rebalancing_policy_index

        ax1.plot(result_times_to_plot[rebalancing_policy], result_values_to_plot[rebalancing_policy], label=rebalancing_policy, linestyle=linestyles[0], color=colors[innermost_index], alpha=1)

        ax1.grid(True)
        # ax1.set_ylim(ymin=0)
        ax1.set_xlabel("Time (minutes)")
        ax1.set_ylabel("Total fortune ($)")

        # Total throughput legend
        lines, labels = ax1.get_legend_handles_labels()
        legend = ax1.legend(lines, labels, loc='best')

        fig.savefig(save_at_directory + filename + "_total_fortune_over_time.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + ".pdf", bbox_inches='tight')

        # legend_filename = filename + "_legend.png"
        # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

    plt.show()


if __name__ == '__main__':
    plot_total_fortune_over_time(filename='results_05')
