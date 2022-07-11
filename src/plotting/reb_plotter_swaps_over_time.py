from math import ceil, floor

from matplotlib.lines import Line2D
from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

# from save_legend import save_legend

outputs_directory = str(Path("../outputs").resolve())
save_at_directory = outputs_directory + "/figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)

filename = 'results_05'

times = 'rebalancing_history_start_times'

traj = load_trajectory(filename=outputs_directory + '/results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_parameters=2, load_results=1)
# traj.v_auto_load = True

# Parse parameter values

par_rebalancing_policy_values = traj.f_get('rebalancing_policy').f_get_range()
par_rebalancing_policy_values = list(dict.fromkeys(par_rebalancing_policy_values))

# Parse results

result_times = list(traj.f_get_from_runs(times, fast_access=True).values())
# result_times = np.reshape(np.array(result_times), (len(par_rebalancing_policy_values), len(result_times)/len(par_rebalancing_policy_values)))

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})
markers = [".", "x", "s", "+", "*", "d"]


fig, ax1 = plt.subplots()
for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
    innermost_index = rebalancing_policy_index

    # ax1.plot(result_times[rebalancing_policy_index], result_values[rebalancing_policy_index], label=rebalancing_policy, linestyle=linestyles[0], marker=markers[innermost_index], color=color, alpha=1)
    ax1.step(result_times[rebalancing_policy_index], np.arange(len(result_times[rebalancing_policy_index])), where='pre', label=rebalancing_policy, linestyle=linestyles[innermost_index], color=colors[innermost_index], alpha=1)



    ax1.grid(True)
    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Number of swaps initiated")

    # Total throughput legend
    lines, labels = ax1.get_legend_handles_labels()
    legend = ax1.legend(lines, labels, loc='best')

    fig.savefig(save_at_directory + filename + "_number_of_swaps.png", bbox_inches='tight')
    # fig.savefig(save_at_directory + filename + ".pdf", bbox_inches='tight')

    # legend_filename = filename + "_legend.png"
    # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

plt.show()
