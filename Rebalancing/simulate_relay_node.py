from numpy import random, round, recfromcsv
import simpy
import sys
import pandas as pd

from Rebalancing.Node import Node
from Rebalancing.Transaction import Transaction


def transaction_generator(env, topology, source, destination, total_transactions, exp_mean, amount_distribution, amount_distribution_parameters, all_transactions_list, verbose):
    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        if amount_distribution == "constant":
            amount = amount_distribution_parameters[0]
        elif amount_distribution == "uniform":
            max_transaction_amount = amount_distribution_parameters[0]
            amount = random.randint(1, max_transaction_amount)
        elif amount_distribution == "gaussian":
            max_transaction_amount = amount_distribution_parameters[0]
            gaussian_mean = amount_distribution_parameters[1]
            gaussian_variance = amount_distribution_parameters[2]
            amount = round(max(1, min(max_transaction_amount, random.normal(gaussian_mean, gaussian_variance))))
        # elif amount_distribution == "pareto":
        #     lower = amount_distribution_parameters[0]  # the lower end of the support
        #     shape = amount_distribution_parameters[1]  # the distribution shape parameter, also known as `a` or `alpha`
        #     size = amount_distribution_parameters[2]  # the size of your sample (number of random values)
        #     amount = random.pareto(shape, size) + lower
        # elif amount_distribution == "powerlaw":
        #     powerlaw.Power_Law(xmin=1, xmax=2, discrete=True, parameters=[1.16]).generate_random(n=10)
        # elif amount_distribution == "empirical_from_csv_file":
        #     dataset = amount_distribution_parameters[0]
        #     data_size = amount_distribution_parameters[1]
        #     amount = dataset[random.randint(0, data_size)]
        else:
            print("Input error: {} is not a supported amount distribution or the parameters {} given are invalid.".format(amount_distribution, amount_distribution_parameters))
            sys.exit(1)

        t = Transaction(env, topology, env.now, source, destination, amount, verbose)
        all_transactions_list.append(t)
        env.process(t.run())

        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)


def simulate_relay_node(node_parameters, experiment_parameters, rebalancing_parameters):

    # initial_balance_L = node_parameters["initial_balance_L"]
    # initial_balance_R = node_parameters["initial_balance_R"]
    # capacity_L = node_parameters["capacity_L"]
    # capacity_R = node_parameters["capacity_R"]
    # fees = [node_parameters["base_fee"], node_parameters["proportional_fee"]]

    total_transactions_L_to_R = experiment_parameters["total_transactions_L_to_R"]
    exp_mean_L_to_R = experiment_parameters["exp_mean_L_to_R"]
    amount_distribution_L_to_R = experiment_parameters["amount_distribution_L_to_R"]
    amount_distribution_parameters_L_to_R = experiment_parameters["amount_distribution_parameters_L_to_R"]

    total_transactions_R_to_L = experiment_parameters["total_transactions_R_to_L"]
    exp_mean_R_to_L = experiment_parameters["exp_mean_R_to_L"]
    amount_distribution_R_to_L = experiment_parameters["amount_distribution_R_to_L"]
    amount_distribution_parameters_R_to_L = experiment_parameters["amount_distribution_parameters_R_to_L"]

    demand_estimates = {"L": 0, "R": 0}
    if amount_distribution_L_to_R == "constant":
        demand_estimates["L"] = exp_mean_L_to_R * amount_distribution_parameters_L_to_R[0]
    elif amount_distribution_L_to_R == "uniform":
        demand_estimates["L"] = exp_mean_L_to_R * amount_distribution_parameters_L_to_R[0] / 2
    elif amount_distribution_L_to_R == "gaussian":
        demand_estimates["L"] = exp_mean_L_to_R * amount_distribution_parameters_L_to_R[1]
    else:
        print("Input error: {} is not a supported amount distribution.".format(amount_distribution_L_to_R))
        sys.exit(1)

    if amount_distribution_R_to_L == "constant":
        demand_estimates["R"] = exp_mean_R_to_L * amount_distribution_parameters_R_to_L[0]
    elif amount_distribution_R_to_L == "uniform":
        demand_estimates["R"] = exp_mean_R_to_L * amount_distribution_parameters_R_to_L[0] / 2
    elif amount_distribution_R_to_L == "gaussian":
        demand_estimates["R"] = exp_mean_R_to_L * amount_distribution_parameters_R_to_L[1]
    else:
        print("Input error: {} is not a supported amount distribution.".format(amount_distribution_R_to_L))
        sys.exit(1)

    verbose = experiment_parameters["verbose"]
    seed = experiment_parameters["seed"]

    # if amount_distribution_L_to_R == "empirical_from_csv_file":
    #     amount_distribution_parameters_L_to_R = [amount_distribution_parameters_L_to_R, len(amount_distribution_parameters_L_to_R)]
    # if amount_distribution_R_to_L == "empirical_from_csv_file":
    #     amount_distribution_parameters_R_to_L = [amount_distribution_parameters_R_to_L, len(amount_distribution_parameters_R_to_L)]

    # total_simulation_time_estimation = max(total_transactions_L_to_R * 1 / exp_mean_L_to_R, total_transactions_R_to_L * 1 / exp_mean_R_to_L)
    total_simulation_time_estimation = min(total_transactions_L_to_R * 1 / exp_mean_L_to_R, total_transactions_R_to_L * 1 / exp_mean_R_to_L)
    random.seed(seed)

    env = simpy.Environment()

    N = Node(env, node_parameters, rebalancing_parameters, demand_estimates, verbose)
    env.process(N.run())

    topology = {"N": N}

    all_transactions_list = []
    env.process(transaction_generator(env, topology, "L", "R", total_transactions_L_to_R, exp_mean_L_to_R, amount_distribution_L_to_R, amount_distribution_parameters_L_to_R, all_transactions_list, verbose))
    env.process(transaction_generator(env, topology, "R", "L", total_transactions_R_to_L, exp_mean_R_to_L, amount_distribution_R_to_L, amount_distribution_parameters_R_to_L, all_transactions_list, verbose))

    # env.run(until=total_simulation_time_estimation + rebalancing_parameters["check_interval"])
    env.run(until=total_simulation_time_estimation)

    # Calculate results

    measurement_interval = [total_simulation_time_estimation*0.1, total_simulation_time_estimation*0.9]

    success_count_L_to_R = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "SUCCEEDED")))
    success_count_R_to_L = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "SUCCEEDED")))
    success_count_node_total = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status == "SUCCEEDED")))
    arrived_count_L_to_R = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status != "PENDING")))
    arrived_count_R_to_L = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status != "PENDING")))
    arrived_count_node_total = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status != "PENDING")))
    success_amount_L_to_R = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "SUCCEEDED")))
    success_amount_R_to_L = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "SUCCEEDED")))
    success_amount_node_total = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status == "SUCCEEDED")))
    arrived_amount_L_to_R = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status != "PENDING")))
    arrived_amount_R_to_L = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status != "PENDING")))
    arrived_amount_node_total = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status != "PENDING")))
    success_rate_L_to_R = success_count_L_to_R/arrived_count_L_to_R
    success_rate_R_to_L = success_count_R_to_L/arrived_count_R_to_L
    success_rate_node_total = success_count_node_total / arrived_count_node_total
    normalized_throughput_L_to_R = success_amount_L_to_R/arrived_amount_L_to_R      # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out
    normalized_throughput_R_to_L = success_amount_R_to_L/arrived_amount_R_to_L      # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out
    normalized_throughput_node_total = success_amount_node_total/arrived_amount_node_total     # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out

    initial_fortune = node_parameters["initial_balance_L"] + node_parameters["initial_balance_R"] + node_parameters["on_chain_budget"]
    final_fortune_without_pending_swaps = N.balances["L"] + N.balances["R"] + N.on_chain_budget
    final_fortune_with_pending_swaps = N.balances["L"] + N.balances["R"] + N.on_chain_budget + N.swap_IN_amounts_in_progress["L"] + N.swap_IN_amounts_in_progress["R"] + N.swap_OUT_amounts_in_progress["L"] + N.swap_OUT_amounts_in_progress["R"]

    for t in all_transactions_list:
        del t.env
        del t.cleared
    all_transactions_list = pd.DataFrame([vars(t) for t in all_transactions_list])

    # all_transaction_signatures = [t.get_transaction_signature() for t in all_transactions_list]
    #
    # # all_transaction_signatures = []
    # # for t in all_transactions_list:
    # #     transaction_signature = t.get_transaction_signature()
    # #     all_transaction_signatures.append(transaction_signature)


    results = {
        'node_parameters': node_parameters,
        'experiment_parameters': experiment_parameters,
        'rebalancing_parameters': rebalancing_parameters,
        'measurement_interval_length': measurement_interval[1] - measurement_interval[0],
        'success_count_L_to_R': success_count_L_to_R,
        'success_count_R_to_L': success_count_R_to_L,
        'success_count_node_total': success_count_node_total,
        'arrived_count_L_to_R': arrived_count_L_to_R,
        'arrived_count_R_to_L': arrived_count_R_to_L,
        'arrived_count_node_total': arrived_count_node_total,
        'success_amount_L_to_R': success_amount_L_to_R,
        'success_amount_R_to_L': success_amount_R_to_L,
        'success_amount_node_total': success_amount_node_total,
        'arrived_amount_L_to_R': arrived_amount_L_to_R,
        'arrived_amount_R_to_L': arrived_amount_R_to_L,
        'arrived_amount_node_total': arrived_amount_node_total,
        'success_rate_L_to_R': success_rate_L_to_R,
        'success_rate_R_to_L': success_rate_R_to_L,
        'success_rate_node_total': success_rate_node_total,
        'normalized_throughput_L_to_R': normalized_throughput_L_to_R,
        'normalized_throughput_R_to_L': normalized_throughput_R_to_L,
        'normalized_throughput_node_total': normalized_throughput_node_total,
        'all_transactions_list': all_transactions_list,
        # 'all_transaction_signatures': all_transaction_signatures
        'initial_fortune': initial_fortune,
        'final_fortune_without_pending_swaps': final_fortune_without_pending_swaps,
        'final_fortune_with_pending_swaps': final_fortune_with_pending_swaps
    }

    print("Initial total fortune of node N = {}".format(initial_fortune))
    print("Final total fortune of node N without pending swaps = {}".format(final_fortune_without_pending_swaps))
    print("Final total fortune of node N with pending swaps = {}\n".format(final_fortune_with_pending_swaps))

    return results

# if __name__ == '__main__':
#     simulate_relay_node()
