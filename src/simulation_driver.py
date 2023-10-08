import pypet
from simulate_relay_node import *


def pypet_wrapper(traj):
    node_parameters = {
        "initial_balance_L": traj.initial_balance_L,
        "initial_balance_R": traj.initial_balance_R,
        "capacity_L": traj.capacity_L,
        "capacity_R": traj.capacity_R,
        "base_fee": traj.base_fee,
        "proportional_fee": traj.proportional_fee,
        "on_chain_budget": traj.on_chain_budget
    }

    experiment_parameters = {
        "total_transactions_L_to_R": traj.total_transactions_L_to_R,
        "exp_mean_L_to_R": traj.exp_mean_L_to_R,
        "amount_distribution_L_to_R": traj.amount_distribution_L_to_R,
        "amount_distribution_parameters_L_to_R": traj.amount_distribution_parameters_L_to_R,
        "total_transactions_R_to_L": traj.total_transactions_R_to_L,
        "exp_mean_R_to_L": traj.exp_mean_R_to_L,
        "amount_distribution_R_to_L": traj.amount_distribution_R_to_L,
        "amount_distribution_parameters_R_to_L": traj.amount_distribution_parameters_R_to_L,
        "verbose": traj.verbose,
        "verbose_also_print_transactions": traj.verbose_also_print_transactions,
        "filename": traj.filename,
        "seed": traj.seed
    }

    rebalancing_parameters = {
        # "server_min_swap_amount": traj.server_min_swap_amount,
        "server_swap_fee": traj.server_swap_fee,
        "rebalancing_policy": traj.rebalancing_policy,
        "autoloop_lower_threshold": traj.autoloop_lower_threshold,
        "autoloop_upper_threshold": traj.autoloop_upper_threshold,
        # "default_swap_amount": traj.default_swap_amount,
        "check_interval": traj.check_interval,
        "T_conf": traj.T_conf,
        "miner_fee": traj.miner_fee,
        "safety_margins_in_minutes": {"L": traj.safety_margin_in_minutes_L, "R": traj.safety_margin_in_minutes_R}
    }

    results = simulate_relay_node(node_parameters, experiment_parameters, rebalancing_parameters)

    traj.f_add_result('measurement_interval_length', results['measurement_interval_length'], comment='Measurement interval length')
    traj.f_add_result('success_count_L_to_R', results['success_count_L_to_R'], comment='Number of successful transactions from L to R')
    traj.f_add_result('success_count_R_to_L', results['success_count_R_to_L'], comment='Number of successful transactions from R to L')
    traj.f_add_result('success_count_node_total', results['success_count_node_total'], comment='Number of successful transactions (node total)')
    traj.f_add_result('failure_count_L_to_R', results['failure_count_L_to_R'], comment='Number of failed transactions from L to R')
    traj.f_add_result('failure_count_R_to_L', results['failure_count_R_to_L'], comment='Number of failed transactions from R to L')
    traj.f_add_result('failure_count_node_total', results['failure_count_node_total'], comment='Number of failed transactions (node total)')
    traj.f_add_result('arrived_count_L_to_R', results['arrived_count_L_to_R'], comment='Number of transactions that arrived from L to R')
    traj.f_add_result('arrived_count_R_to_L', results['arrived_count_R_to_L'], comment='Number of transactions that arrived from R to L')
    traj.f_add_result('arrived_count_node_total', results['arrived_count_node_total'], comment='Number of transactions that arrived (node total)')
    traj.f_add_result('success_amount_L_to_R', results['success_amount_L_to_R'], comment='Throughput (Amount of successful transactions) from L to R')
    traj.f_add_result('success_amount_R_to_L', results['success_amount_R_to_L'], comment='Throughput (Amount of successful transactions) from R to L')
    traj.f_add_result('success_amount_node_total', results['success_amount_node_total'], comment='Throughput (Amount of successful transactions) (node total)')
    traj.f_add_result('failure_amount_L_to_R', results['failure_amount_L_to_R'], comment='Blockage (Amount of failed transactions) from L to R')
    traj.f_add_result('failure_amount_R_to_L', results['failure_amount_R_to_L'], comment='Blockage (Amount of failed transactions) from R to L')
    traj.f_add_result('failure_amount_node_total', results['failure_amount_node_total'], comment='Blockage (Amount of failed transactions) (node total)')
    traj.f_add_result('arrived_amount_L_to_R', results['arrived_amount_L_to_R'], comment='Amount of transactions that arrived from L to R')
    traj.f_add_result('arrived_amount_R_to_L', results['arrived_amount_R_to_L'], comment='Amount of transactions that arrived from R to L')
    traj.f_add_result('arrived_amount_node_total', results['arrived_amount_node_total'], comment='Amount of transactions that arrived (node total)')
    traj.f_add_result('success_rate_L_to_R', results['success_rate_L_to_R'], comment='Success rate from L to R')
    traj.f_add_result('success_rate_R_to_L', results['success_rate_R_to_L'], comment='Success rate from R to L')
    traj.f_add_result('success_rate_node_total', results['success_rate_node_total'], comment='Success rate (node total)')
    traj.f_add_result('normalized_throughput_L_to_R', results['normalized_throughput_L_to_R'], comment='Normalized throughput from L to R')
    traj.f_add_result('normalized_throughput_R_to_L', results['normalized_throughput_R_to_L'], comment='Normalized throughput from R to L')
    traj.f_add_result('normalized_throughput_node_total', results['normalized_throughput_node_total'], comment='Normalized throughput (node total)')
    traj.f_add_result('initial_fortune', results['initial_fortune'], comment='Initial fortune of node')
    traj.f_add_result('final_fortune_without_pending_swaps', results['final_fortune_without_pending_swaps'], comment='Final total fortune of node N without pending swaps')
    traj.f_add_result('final_fortune_with_pending_swaps', results['final_fortune_with_pending_swaps'], comment='Final total fortune of node N with pending swaps')
    traj.f_add_result('final_fortune_with_pending_swaps_minus_losses', results['final_fortune_with_pending_swaps_minus_losses'], comment='Final total fortune of node N with pending swaps minus fee losses')

    # traj.f_add_result('all_transactions_list', results['all_transactions_list'], 'All transactions')
    # traj.f_add_result('all_transaction_signatures', results['all_transaction_signatures'], 'All transactions')

    traj.f_add_result('balance_history_times', results['balance_history_times'], comment='Balance history times')
    traj.f_add_result('balance_history_values_L', results['balance_history_values_L'], comment='Balance history values for channel N-L')
    traj.f_add_result('balance_history_values_R', results['balance_history_values_R'], comment='Balance history values for channel N-R')
    traj.f_add_result('total_fortune_including_pending_swaps_times', results['total_fortune_including_pending_swaps_times'], comment='Times for total fortune of node N including pending swaps')
    traj.f_add_result('total_fortune_including_pending_swaps_values', results['total_fortune_including_pending_swaps_values'], comment='Values for total fortune of node N including pending swaps')
    traj.f_add_result('total_fortune_including_pending_swaps_minus_losses_values', results['total_fortune_including_pending_swaps_minus_losses_values'], comment='Values for total fortune of node N including pending swaps minus fee losses')
    traj.f_add_result('cumulative_fee_losses', results['cumulative_fee_losses'], comment='Cumulative fee losses of node N from dropped transactions for the entire duration of the simulation')
    traj.f_add_result('cumulative_rebalancing_fees', results['cumulative_rebalancing_fees'], comment='Cumulative fees paid by node N for rebalancing operations for the entire duration of the simulation')
    traj.f_add_result('fee_losses_over_time', results['fee_losses_over_time'], comment='Fee losses of node N from dropped transactions measured at times of transaction processing')
    traj.f_add_result('rebalancing_fees_over_time', results['rebalancing_fees_over_time'], comment='Rebalancing fees paid by node N measured at times of transaction processing')

    traj.f_add_result('rebalancing_history_start_times', results['rebalancing_history_start_times'], comment='Swap start times')
    traj.f_add_result('rebalancing_history_end_times', results['rebalancing_history_end_times'], comment='Swap end times')
    traj.f_add_result('rebalancing_history_types', results['rebalancing_history_types'], comment='Swap types')
    traj.f_add_result('rebalancing_history_amounts', results['rebalancing_history_amounts'], comment='Swap amounts')
    traj.f_add_result('rebalancing_history_results', results['rebalancing_history_results'], comment='Swap results (success/failure)')


def main():
    # SIMULATION PARAMETERS
    filename = 'results_test'

    # verbose = True
    verbose = False
    verbose_also_print_transactions = False
    # verbose_also_print_transactions = True
    num_of_experiments = 1

    base_fee = 0    # 4e-4
    # proportional_fee = 3e-5
    proportional_fee = 0.01
    on_chain_budget = 1000

    # Channel N-L
    initial_balance_L = 500
    capacity_L = 1000
    total_transactions_L_to_R = 60000
    exp_mean_L_to_R = 10 / 1     # transactions per minute
    # amount_distribution_L_to_R = "constant"
    # amount_distribution_parameters_L_to_R = [30]                  # value of all transactions
    # amount_distribution_L_to_R = "uniform"
    # amount_distribution_parameters_L_to_R = [50]                # max_transaction_amount
    amount_distribution_L_to_R = "gaussian"
    amount_distribution_parameters_L_to_R = [100, 25, 20]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]

    # Channel N-R
    initial_balance_R = 500
    capacity_R = 1000
    total_transactions_R_to_L = 15000
    exp_mean_R_to_L = 2.5 / 1     # transactions per minute
    # amount_distribution_R_to_L = "constant"
    # amount_distribution_parameters_R_to_L = [30]                   # value of all transactions
    # amount_distribution_R_to_L = "uniform"
    # amount_distribution_parameters_R_to_L = [50]                # max_transaction_amount
    amount_distribution_R_to_L = "gaussian"
    amount_distribution_parameters_R_to_L = [100, 25, 20]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]

    # REBALANCING
    # LSP parameters
    # server_min_swap_amount = 100
    server_swap_fee = 0.005      # fraction of swap amount

    # Node parameters
    # rebalancing_policy = "None"
    # rebalancing_policy = "Autoloop"
    rebalancing_policy = "Loopmax"
    # rebalancing_policy_parameters = [0.2, 0.8, server_min_swap_amount]  # [min % balance, max % balance, margin from target to launch]
    autoloop_lower_threshold = 0.3
    autoloop_upper_threshold = 0.7
    # default_swap_amount = server_min_swap_amount
    check_interval = 10     # minutes

    T_conf = 9.99     # minutes
    miner_fee = 2

    safety_margin_in_minutes_L = T_conf/5
    safety_margin_in_minutes_R = T_conf/5

    # Create the environment
    env = pypet.Environment(trajectory='relay_node_channel_rebalancing',
                            filename='../outputs/results/' + filename + '.hdf5',
                            log_folder='../outputs/logs/',
                            log_stdout=True,
                            overwrite_file=True)
    traj = env.traj

    # Encode parameters for pypet

    traj.f_add_parameter('initial_balance_L', initial_balance_L, comment='Initial balance of node N in channel with node L')
    traj.f_add_parameter('capacity_L', capacity_L, comment='Capacity of channel N-L')
    traj.f_add_parameter('total_transactions_L_to_R', total_transactions_L_to_R, comment='Total transactions from L to R')
    traj.f_add_parameter('exp_mean_L_to_R', exp_mean_L_to_R, comment='Rate of exponentially distributed arrivals from L to R')
    traj.f_add_parameter('amount_distribution_L_to_R', amount_distribution_L_to_R, comment='The distribution of the transaction amounts from L to R')
    traj.f_add_parameter('amount_distribution_parameters_L_to_R', amount_distribution_parameters_L_to_R, comment='Parameters of the distribution of the transaction amounts from L to R')

    traj.f_add_parameter('initial_balance_R', initial_balance_R, comment='Initial balance of node N in channel with node R')
    traj.f_add_parameter('capacity_R', capacity_R, comment='Capacity of channel N-R')
    traj.f_add_parameter('total_transactions_R_to_L', total_transactions_R_to_L, comment='Total transactions from R to L')
    traj.f_add_parameter('exp_mean_R_to_L', exp_mean_R_to_L, comment='Rate of exponentially distributed arrivals from R to L')
    traj.f_add_parameter('amount_distribution_R_to_L', amount_distribution_R_to_L, comment='The distribution of the transaction amounts from R to L')
    traj.f_add_parameter('amount_distribution_parameters_R_to_L', amount_distribution_parameters_R_to_L, comment='Parameters of the distribution of the transaction amounts from R to L')

    traj.f_add_parameter('base_fee', base_fee, comment='Base forwarding fee charged by node N')
    traj.f_add_parameter('proportional_fee', proportional_fee, comment='Proportional forwarding fee charged by node N')
    traj.f_add_parameter('on_chain_budget', on_chain_budget, comment='On-chain budget of node N')

    # traj.f_add_parameter('server_min_swap_amount', server_min_swap_amount, comment='Minimum amount the LSP allows for a swap')
    traj.f_add_parameter('server_swap_fee', server_swap_fee, comment='Percentage of swap amount the LSP charges as fees')
    traj.f_add_parameter('rebalancing_policy', rebalancing_policy, comment='Rebalancing policy')
    traj.f_add_parameter('autoloop_lower_threshold', autoloop_lower_threshold, comment='Balance percentage threshold below which the channel needs a swap-in according to the Autoloop policy')
    traj.f_add_parameter('autoloop_upper_threshold', autoloop_upper_threshold, comment='Balance percentage threshold above which the channel needs a swap-out according to the Autoloop policy')
    # traj.f_add_parameter('default_swap_amount', default_swap_amount, comment='Default swap amount node N requests')
    traj.f_add_parameter('check_interval', check_interval, comment='Time in seconds every which a check for rebalancing is performed')
    traj.f_add_parameter('T_conf', T_conf, comment='Confirmation time (seconds) for an on-chain transaction')
    traj.f_add_parameter('miner_fee', miner_fee, comment='Miner fee for an on-chain transaction')
    traj.f_add_parameter('safety_margin_in_minutes_L', safety_margin_in_minutes_L, comment='Safety margin in minutes for swaps under the Loopmax policy for node L')
    traj.f_add_parameter('safety_margin_in_minutes_R', safety_margin_in_minutes_R, comment='Safety margin in minutes for swaps under the Loopmax policy for node R')

    # traj.f_add_parameter('verbose', verbose, comment='Verbose output')
    traj.f_add_parameter('verbose', verbose, comment='Verbose output at rebalancing check times')
    traj.f_add_parameter('verbose_also_print_transactions', verbose_also_print_transactions, comment='Verbose output for all transactions apart from rebalancing check times')
    traj.f_add_parameter('filename', filename, comment='Filename of the results')
    traj.f_add_parameter('num_of_experiments', num_of_experiments, comment='Repetitions of every experiment')
    traj.f_add_parameter('seed', 0, comment='Randomness seed')

    seeds = [63621, 87563, 24240, 14020, 84331, 60917, 48692, 73114, 90695, 62302, 52578, 43760, 84941, 30804, 40434, 63664, 25704, 38368, 45271, 34425]

    traj.f_explore(pypet.cartesian_product({
        # 'base_fee': [float(4*10**P) for P in list(range(-4, 2))],
        # 'base_fee': [40.0],
        # 'proportional_fee': [float(10**P) for P in list(range(-5, 0))],
        # 'proportional_fee': [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1],        # <-----
        # 'proportional_fee': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
        # 'proportional_fee': [0.01, 0.05, 0.1, 0.2, 0.25],
        # 'proportional_fee': [0.1],
        # 'proportional_fee': [0.02],
        # 'autoloop_upper_threshold': [0.7, 0.8, 0.9, 0.95, 0.99],
        # 'autoloop_upper_threshold': [0.7],
        # 'rebalancing_policy': ['None', 'Autoloop', 'Loopmax'],
        'rebalancing_policy': ['None', 'Autoloop', 'Loopmax', 'RebEL'],
        # 'rebalancing_policy': ['RebEL'],
        # 'rebalancing_policy': ['Autoloop', 'RebEL'],
        # 'rebalancing_policy': ['Autoloop'],
        # 'rebalancing_policy': ['Loopmax'],
        # 'rebalancing_policy': ['None'],
        # 'rebalancing_policy': ['Autoloop-infrequent'],
        # 'rebalancing_policy': ['None', 'Autoloop', 'Autoloop-infrequent', 'Loopmax'],
        'seed': seeds[0:traj.num_of_experiments]
    }))

    # Run wrapping function instead of simulator directly
    env.run(pypet_wrapper)
    env.disable_logging()


if __name__ == '__main__':
    main()
