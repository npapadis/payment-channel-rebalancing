import pypet
from reb_simulate_channel import *
import csv
from statsmodels.distributions.empirical_distribution import ECDF
from math import floor, ceil


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
        "seed": traj.seed
    }

    rebalancing_parameters = {
        "server_min_swap_amount": traj.server_min_swap_amount,
        "server_swap_fee": traj.server_swap_fee,
        "rebalancing_policy": traj.rebalancing_policy,
        "lower_threshold": traj.lower_threshold,
        "upper_threshold": traj.upper_threshold,
        "swap_amount": traj.swap_amount,
        "T_conf": traj.T_conf,
        "miner_fee": traj.miner_fee
    }

    results, all_transactions_list = simulate_node(node_parameters, experiment_parameters, rebalancing_parameters)

    # traj.f_add_result('measurement_interval_length', results['measurement_interval_length'], comment='Measurement interval length')
    # traj.f_add_result('success_count_node_0', results['success_counts'][0], comment='Number of successful transactions (node 0)')
    # traj.f_add_result('success_count_node_1', results['success_counts'][1], comment='Number of successful transactions (node 1)')
    # traj.f_add_result('success_count_channel_total', results['success_counts'][2], comment='Number of successful transactions (channel total)')
    # traj.f_add_result('arrived_count_node_0', results['arrived_counts'][0], comment='Number of transactions that arrived (node 0)')
    # traj.f_add_result('arrived_count_node_1', results['arrived_counts'][1], comment='Number of transactions that arrived (node 1)')
    # traj.f_add_result('arrived_count_channel_total', results['arrived_counts'][2], comment='Number of transactions that arrived (channel total)')
    # traj.f_add_result('success_amount_node_0', results['success_amounts'][0], comment='Throughput (Amount of successful transactions) (node 0)')
    # traj.f_add_result('success_amount_node_1', results['success_amounts'][1], comment='Throughput (Amount of successful transactions) (node 1)')
    # traj.f_add_result('success_amount_channel_total', results['success_amounts'][2], comment='Throughput (Amount of successful transactions) (channel total)')
    # traj.f_add_result('arrived_amount_node_0', results['arrived_amounts'][0], comment='Amount of transactions that arrived (node 0)')
    # traj.f_add_result('arrived_amount_node_1', results['arrived_amounts'][1], comment='Amount of transactions that arrived (node 1)')
    # traj.f_add_result('arrived_amount_channel_total', results['arrived_amounts'][2], comment='Amount of transactions that arrived (channel total)')
    # traj.f_add_result('sacrificed_count_node_0', results['sacrificed_counts'][0], comment='Number of sacrificed transactions (node 0)')
    # traj.f_add_result('sacrificed_count_node_1', results['sacrificed_counts'][1], comment='Number of sacrificed transactions (node 1)')
    # traj.f_add_result('sacrificed_count_channel_total', results['sacrificed_counts'][2], comment='Number of sacrificed transactions (channel total)')
    # traj.f_add_result('sacrificed_amount_node_0', results['sacrificed_amounts'][0], comment='Amount of sacrificed transactions (node 0)')
    # traj.f_add_result('sacrificed_amount_node_1', results['sacrificed_amounts'][1], comment='Amount of sacrificed transactions (node 1)')
    # traj.f_add_result('sacrificed_amount_channel_total', results['sacrificed_amounts'][2], comment='Amount of sacrificed transactions (channel total)')
    # traj.f_add_result('success_rate_node_0', results['success_rates'][0], comment='Success rate (node 0)')
    # traj.f_add_result('success_rate_node_1', results['success_rates'][1], comment='Success rate (node 1)')
    # traj.f_add_result('success_rate_channel_total', results['success_rates'][2], comment='Success rate (channel total)')
    # traj.f_add_result('normalized_throughput_node_0', results['normalized_throughputs'][0], comment='Normalized throughput (node 0)')
    # traj.f_add_result('normalized_throughput_node_1', results['normalized_throughputs'][1], comment='Normalized throughput (node 1)')
    # traj.f_add_result('normalized_throughput_channel_total', results['normalized_throughputs'][2], comment='Normalized throughput (channel total)')
    # traj.f_add_result('total_queueing_time_of_successful_transactions', results['total_queueing_times'][0], comment='Total queueing time of successful transactions')
    # traj.f_add_result('total_queueing_time_of_all_transactions', results['total_queueing_times'][1], comment='Total queueing time of all transactions')
    # traj.f_add_result('average_total_queueing_time_per_successful_unit_amount', results['total_queueing_times'][2], comment='Average queueing delay per successful unit amount')
    # traj.f_add_result('average_total_queueing_time_per_successful_transaction', results['total_queueing_times'][3], comment='Average queueing delay per transaction')

    # traj.f_add_result('all_transactions_list', all_transactions_list, 'All transactions')



def main():
    # Create the environment
    env = pypet.Environment(trajectory='single_payment_channel_scheduling',
                            filename='../HDF5/results_100.hdf5',
                            overwrite_file=True)
    traj = env.traj
    # EMPIRICAL_DATA_FILEPATH = "./creditcard-non-fraudulent-only-amounts-only.csv"

    # SIMULATION PARAMETERS

    verbose = True
    num_of_experiments = 1

    base_fee = 10.0
    proportional_fee = 0.1
    on_chain_budget = 1000

    # Channel N-L
    initial_balance_L = 0
    capacity_L = 300
    total_transactions_L_to_R = 500
    exp_mean_L_to_R = 1 / 3
    amount_distribution_L_to_R = "constant"
    amount_distribution_parameters_L_to_R = [100]                  # value of all transactions
    # amount_distribution_L_to_R = "uniform"
    # amount_distribution_parameters_L_to_R = [100]                # max_transaction_amount
    # amount_distribution_L_to_R = "gaussian"
    # amount_distribution_parameters_L_to_R = [300, 100, 50]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]

    # Channel N-R
    initial_balance_R = 300         # Capacity = 300
    capacity_R = 300
    total_transactions_R_to_L = 500
    exp_mean_R_to_L = 1 / 3
    amount_distribution_R_to_L = "constant"
    amount_distribution_parameters_R_to_L = [100]                  # value of all transactions
    # amount_distribution_R_to_L = "uniform"
    # amount_distribution_parameters_R_to_L = [100]                # max_transaction_amount
    # amount_distribution_R_to_L = "gaussian"
    # amount_distribution_parameters_R_to_L = [300, 100, 50]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]

    # REBALANCING
    # LSP parameters
    server_min_swap_amount = 20
    server_swap_fee = 0.05      # percentage of swap amount

    # Node parameters
    rebalancing_policy = "autoloop"
    # rebalancing_policy_parameters = [0.2, 0.8, server_min_swap_amount]  # [min % balance, max % balance, margin from target to launch]
    lower_threshold = 0.2
    upper_threshold = 0.8
    swap_amount = server_min_swap_amount

    T_conf = 60
    miner_fee = 10

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

    traj.f_add_parameter('server_min_swap_amount', server_min_swap_amount, comment='Minimum amount the LSP allows for a swap')
    traj.f_add_parameter('server_swap_fee', server_swap_fee, comment='Percentage of swap amount the LSP charges as fees')
    traj.f_add_parameter('rebalancing_policy', rebalancing_policy, comment='Rebalancing policy')
    traj.f_add_parameter('lower_threshold', lower_threshold, comment='Balance percentage threshold below which the channel needs a swap-in')
    traj.f_add_parameter('upper_threshold', upper_threshold, comment='Balance percentage threshold above which the channel needs a swap-out')
    traj.f_add_parameter('swap_amount', swap_amount, comment='Swap amount node N requests')
    traj.f_add_parameter('T_conf', T_conf, comment='Confirmation time for an on-chain transaction')
    traj.f_add_parameter('miner_fee', miner_fee, comment='Miner fee for an on-chain transaction')

    traj.f_add_parameter('verbose', verbose, comment='Verbose output')
    traj.f_add_parameter('num_of_experiments', num_of_experiments, comment='Repetitions of every experiment')
    traj.f_add_parameter('seed', 0, comment='Randomness seed')

    seeds = [63621, 87563, 24240, 14020, 84331, 60917, 48692, 73114, 90695, 62302, 52578, 43760, 84941, 30804, 40434, 63664, 25704, 38368, 45271, 34425]

    traj.f_explore(pypet.cartesian_product({
                                            'seed': seeds[1:traj.num_of_experiments + 1]
                                            }))

    # Run wrapping function instead of simulator directly
    env.run(pypet_wrapper)


if __name__ == '__main__':
    main()
