from math import ceil, floor

from matplotlib.lines import Line2D
from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

# from save_legend import save_legend

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
# filename = 'results_01'
filename = 'results_02'

traj = load_trajectory(filename='./results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./results/' + filename + '.hdf5', name='relay_node_channel_rebalancing', load_parameters=2, load_results=1)
# traj.v_auto_load = True

# Parse parameter values

# fee_studied = 'base_fee'
fee_studied = 'proportional_fee'

par_fee_values = traj.f_get(fee_studied).f_get_range()
par_fee_values = list(dict.fromkeys(par_fee_values))
par_rebalancing_policy_values = traj.f_get('rebalancing_policy').f_get_range()
par_rebalancing_policy_values = list(dict.fromkeys(par_rebalancing_policy_values))
par_seed_values = traj.f_get('seed').f_get_range()
par_seed_values = list(dict.fromkeys(par_seed_values))

# Parse results

# result = 'final_fortune_with_pending_swaps'
result = 'success_rate_node_total'

result_values = list(traj.f_get_from_runs(result, fast_access=True).values())
result_values = np.reshape(np.array(result_values), (len(par_fee_values), len(par_rebalancing_policy_values), len(par_seed_values)))
result_values_average = result_values.mean(axis=2)
result_values_max = result_values.max(axis=2)
result_values_min = result_values.min(axis=2)


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})
markers = [".", "x", "s", "+", "*", "d"]


fig, ax1 = plt.subplots()
for rebalancing_policy_index, rebalancing_policy in enumerate(par_rebalancing_policy_values):
    innermost_index = rebalancing_policy_index
    color = colors[innermost_index]

    # Total throughput
    ax1.plot(par_fee_values, result_values_average[:, rebalancing_policy_index], label=rebalancing_policy, linestyle=linestyles[0], marker=markers[innermost_index], color=color, alpha=1)
    # yerr_total = [result_values_average[:, rebalancing_policy_index] - result_values_min[:, rebalancing_policy_index], result_values_max[:, rebalancing_policy_index] - result_values_average[:, rebalancing_policy_index]]
    # ax1.errorbar(par_proportional_fee_values, result_values_average, yerr=yerr_total, fmt='none')

    ax1.set_xscale('log')
    ax1.grid(True)
    # ax1.set_ylim(bottom=0)
    ax1.set_xlabel(fee_studied + " value")
    # ax1.set_ylabel("Total final fortune ($)")

    # Total throughput legend
    lines, labels = ax1.get_legend_handles_labels()
    legend = ax1.legend(lines, labels, loc='best')

    fig.savefig(save_at_directory + filename + "_" + result + ".png", bbox_inches='tight')
    # fig.savefig(save_at_directory + filename + ".pdf", bbox_inches='tight')

    # legend_filename = filename + "_legend.png"
    # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

plt.show()
