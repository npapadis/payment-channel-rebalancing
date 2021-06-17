from numpy import random, round, recfromcsv
import simpy
import sys
# import powerlaw
import pandas as pd
import sortedcontainers as sc
from fractions import Fraction
from math import inf


class Node:
    def __init__(self, env, node_parameters, rebalancing_parameters, verbose):
        self.env = env
        self.balances = {"L": node_parameters["initial_balance_L"], "R": node_parameters["initial_balance_R"]}
        self.capacities = {"L": node_parameters["capacity_L"], "R": node_parameters["capacity_R"]}
        self.fees = [node_parameters["base_fee"], node_parameters["proportional_fee"]]
        self.on_chain_budget = node_parameters["on_chain_budget"]
        self.rebalancing_parameters = rebalancing_parameters
        self.verbose = verbose

        self.node_processor = simpy.Resource(env, capacity=1)
        self.rebalancing_locks = {"L": simpy.Resource(env, capacity=1), "R": simpy.Resource(env, capacity=1)}     # max 1 rebalancing operation active at a time
        self.rebalance_requests = {"L": self.env.event(), "R": self.env.event()}

        self.balance_history_times = []
        self.balance_history_values = []
        self.rebalancing_history_start_times = []
        self.rebalancing_history_end_times = []
        self.rebalancing_history_types = []
        self.rebalancing_history_amounts = []

    def calculate_fees(self, amount):
        return self.fees[0] + amount*self.fees[1]

    def execute_feasible_transaction(self, t):
        # Calling this function requires checking for transaction feasibility beforehand. The function itself does not perform any checks, and this could lead to negative balances if misused.

        self.balances[t.previous_node] += t.amount
        self.balances[t.next_node] -= (t.amount - self.calculate_fees(t.amount))
        self.balance_history_times.append(self.env.now)
        self.balance_history_values.append(self.balances)
        t.status = "SUCCEEDED"

        if self.verbose:
            print("Time {:.2f}: SUCCESS: Transaction {} processed.".format(self.env.now, t))
            print("Time {:.2f}: New balances are {}.".format(self.env.now, self.balances))

    def reject_transaction(self, t):
        t.status = "FAILED"
        if self.verbose:
            print("Time {:.2f}: FAILURE: Transaction {} rejected.".format(self.env.now, t))
            print("Time {:.2f}: Unchanged balances are {}.".format(self.env.now, self.balances))

    def process_transaction(self, t):
        if (t.amount >= self.calculate_fees(t.amount)) and (t.amount <= self.capacities[t.previous_node] - self.balances[t.previous_node]) and (t.amount - self.calculate_fees(t.amount) <= self.balances[t.next_node]):
            self.execute_feasible_transaction(t)
        else:
            self.reject_transaction(t)

    def perform_rebalancing_if_needed(self):
        for neighbor in ["L", "R"]:
            if self.rebalancing_locks[neighbor].count == 0:     # if no rebalancing in progress in the N-neighbor channel
                with self.rebalancing_locks[neighbor].request() as rebalance_request:  # Generate a request event
                    yield rebalance_request

                    if self.rebalancing_parameters["rebalancing_policy"] == "none":
                        pass
                        # return "none"
                    elif self.rebalancing_parameters["rebalancing_policy"] == "autoloop":
                        if self.balances[neighbor] < self.rebalancing_parameters["lower_threshold"] * self.capacities[neighbor]:
                            self.rebalancing_history_start_times.append(self.env.now)
                            self.rebalancing_history_types.append(neighbor + "-IN")
                            self.rebalancing_history_amounts.append(self.rebalancing_parameters["swap_amount"])
                            if self.verbose:
                                print("Time {:.2f}: SWAP-IN initiated in channel N-{} with amount {}.". format(self.env.now, neighbor, self.rebalancing_parameters["swap_amount"]))

                            self.on_chain_budget -= (self.rebalancing_parameters["swap_amount"]*(1+self.rebalancing_parameters["server_swap_fee"]) + self.rebalancing_parameters["miner_fee"])
                            yield self.env.timeout(self.rebalancing_parameters["T_conf"])
                            self.balances[neighbor] += self.rebalancing_parameters["swap_amount"]

                            self.rebalancing_history_end_times.append(self.env.now)
                            if self.verbose:
                                print("Time {:.2f}: SWAP-IN completed in channel N-{} with amount {}.". format(self.env.now, neighbor, self.rebalancing_parameters["swap_amount"]))
                                print("Time {:.2f}: New balances are {}.".format(self.env.now, self.balances))
                            self.rebalance_requests[neighbor].succeed()
                            self.rebalance_requests[neighbor] = self.env.event()
                            # return neighbor + "-in"

                        elif self.balances[neighbor] > self.rebalancing_parameters["upper_threshold"] * self.capacities[neighbor]:
                            self.rebalancing_history_start_times.append(self.env.now)
                            self.rebalancing_history_types.append(neighbor + "-OUT")
                            self.rebalancing_history_amounts.append(self.rebalancing_parameters["swap_amount"])
                            if self.verbose:
                                print("Time {:.2f}: SWAP-OUT initiated in channel N-{} with amount {}.". format(self.env.now, neighbor, self.rebalancing_parameters["swap_amount"]))

                            self.balances[neighbor] -= self.rebalancing_parameters["swap_amount"]
                            yield self.env.timeout(self.rebalancing_parameters["T_conf"])
                            self.on_chain_budget += (self.rebalancing_parameters["swap_amount"]*(1-self.rebalancing_parameters["server_swap_fee"]) - self.rebalancing_parameters["miner_fee"])

                            self.rebalancing_history_end_times.append(self.env.now)
                            if self.verbose:
                                print("Time {:.2f}: SWAP-OUT completed in channel N-{} with amount {}.". format(self.env.now, neighbor, self.rebalancing_parameters["swap_amount"]))
                                print("Time {:.2f}: New balances are {}.".format(self.env.now, self.balances))
                            self.rebalance_requests[neighbor].succeed()
                            self.rebalance_requests[neighbor] = self.env.event()
                            # return neighbor + "-out"

                        else:
                            pass    # no rebalancing needed
                            # return False
            else:
                pass  # if rebalancing already in progress, do not check again if rebalancing is needed

    def run(self):
        while True:
            # yield self.rebalance_requests["L"] | self.rebalance_requests["R"]
            yield self.env.process(self.perform_rebalancing_if_needed())
            yield self.env.timeout(10)

class Transaction:
    # def __init__(self, env, time_of_arrival, path, previous_node, current_node, next_node, amount, verbose):
    def __init__(self, env, topology, time_of_arrival, source, destination, amount, verbose):
        self.env = env
        self.time_of_arrival = time_of_arrival
        self.source = source
        self.destination = destination
        # self.path = path                    # List of strings
        # self.previous_node = previous_node  # String
        # self.current_node = current_node    # Node object
        # self.next_node = next_node          # String
        self.amount = amount
        self.verbose = verbose
        self.status = "PENDING"
        self.pathfinder(topology)

        if self.verbose:
            print("Time {:.2f}: Transaction {} generated.".format(self.env.now, self))

        # Start the run process every time an instance is created.
        # env.process(self.run())

    def pathfinder(self, topology):
        if self.source == "L" and self.destination == "R":
            self.path = ["L", "N", "R"]
            self.previous_node = "L"
            self.current_node = topology["N"]
            self.next_node = "R"
        elif self.source == "R" and self.destination == "L":
            self.path = ["R", "N", "L"]
            self.previous_node = "R"
            self.current_node = topology["N"]
            self.next_node = "L"
        else:
            print("Input error")
            sys.exit(1)

    def run(self):
        with self.current_node.node_processor.request() as process_request:     # Generate a request event
            yield process_request                                               # Wait for access to the node_processor
            self.current_node.process_transaction(self)                         # Once the channel belongs to the transaction, try to process it.

    def __repr__(self):
        return "%s->%s t=%.2f a=%d" % (self.source, self.destination, self.time_of_arrival, self.amount)


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


def simulate_node(node_parameters, experiment_parameters, rebalancing_parameters):

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

    verbose = experiment_parameters["verbose"]
    seed = experiment_parameters["seed"]

    # if amount_distribution_L_to_R == "empirical_from_csv_file":
    #     amount_distribution_parameters_L_to_R = [amount_distribution_parameters_L_to_R, len(amount_distribution_parameters_L_to_R)]
    # if amount_distribution_R_to_L == "empirical_from_csv_file":
    #     amount_distribution_parameters_R_to_L = [amount_distribution_parameters_R_to_L, len(amount_distribution_parameters_R_to_L)]

    # total_simulation_time_estimation = max(total_transactions_L_to_R * 1 / exp_mean_L_to_R, total_transactions_R_to_L * 1 / exp_mean_R_to_L)
    random.seed(seed)

    env = simpy.Environment()

    N = Node(env, node_parameters, rebalancing_parameters, verbose)
    env.process(N.run())

    topology = {"N": N}

    all_transactions_list = []
    env.process(transaction_generator(env, topology, "L", "R", total_transactions_L_to_R, exp_mean_L_to_R, amount_distribution_L_to_R, amount_distribution_parameters_L_to_R, all_transactions_list, verbose))
    env.process(transaction_generator(env, topology, "R", "L", total_transactions_R_to_L, exp_mean_R_to_L, amount_distribution_R_to_L, amount_distribution_parameters_R_to_L, all_transactions_list, verbose))

    env.run()

    # Calculate results

    # measurement_interval = [total_simulation_time_estimation*0.1, total_simulation_time_estimation*0.9]
    #
    # success_count_node_0 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.status == "SUCCEEDED")))
    # success_count_node_1 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.status == "SUCCEEDED")))
    # success_count_channel_total = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status == "SUCCEEDED")))
    # arrived_count_node_0 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.status != "PENDING")))
    # arrived_count_node_1 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.status != "PENDING")))
    # arrived_count_channel_total = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status != "PENDING")))
    # success_amount_node_0 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.status == "SUCCEEDED")))
    # success_amount_node_1 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.status == "SUCCEEDED")))
    # success_amount_channel_total = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status == "SUCCEEDED")))
    # arrived_amount_node_0 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.status != "PENDING")))
    # arrived_amount_node_1 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.status != "PENDING")))
    # arrived_amount_channel_total = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.status != "PENDING")))
    # sacrificed_count_node_0 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # sacrificed_count_node_1 = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # sacrificed_count_channel_total = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # sacrificed_amount_node_0 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 0) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # sacrificed_amount_node_1 = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.from_node == 1) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # sacrificed_amount_channel_total = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.initially_feasible is True) and (t.status in ["REJECTED", "EXPIRED"])))
    # success_rate_node_0 = success_count_node_0/arrived_count_node_0
    # success_rate_node_1 = success_count_node_1/arrived_count_node_1
    # success_rate_channel_total = success_count_channel_total / arrived_count_channel_total
    # normalized_throughput_node_0 = success_amount_node_0/arrived_amount_node_0      # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out
    # normalized_throughput_node_1 = success_amount_node_1/arrived_amount_node_1      # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out
    # normalized_throughput_channel_total = success_amount_channel_total/arrived_amount_channel_total     # should be divided by duration of measurement_interval in both numerator and denominator, but these terms cancel out
    #
    # results = {
    #     'measurement_interval_length': measurement_interval[1] - measurement_interval[0],
    #     'success_counts': [success_count_node_0, success_count_node_1, success_count_channel_total],
    #     'arrived_counts': [arrived_count_node_0, arrived_count_node_1, arrived_count_channel_total],
    #     'success_amounts': [success_amount_node_0, success_amount_node_1, success_amount_channel_total],
    #     'arrived_amounts': [arrived_amount_node_0, arrived_amount_node_1, arrived_amount_channel_total],
    #     'sacrificed_counts': [sacrificed_count_node_0, sacrificed_count_node_1, sacrificed_count_channel_total],
    #     'sacrificed_amounts': [sacrificed_amount_node_0, sacrificed_amount_node_1, sacrificed_amount_channel_total],
    #     'success_rates': [success_rate_node_0, success_rate_node_1, success_rate_channel_total],
    #     'normalized_throughputs': [normalized_throughput_node_0, normalized_throughput_node_1, normalized_throughput_channel_total],
    # }
    #
    # print("Total success rate: {:.2f}".format(success_count_channel_total/arrived_count_channel_total))
    # print("Total normalized throughput: {:.2f}".format(success_amount_channel_total/arrived_amount_channel_total))
    # print("Number of sacrificed transactions (node 0, node 1, total): {}, {}, {}".format(sacrificed_count_node_0, sacrificed_count_node_1, sacrificed_count_channel_total))
    results = {}

    for t in all_transactions_list:
        del t.env

    all_transactions_list = pd.DataFrame([vars(t) for t in all_transactions_list])

    return results, all_transactions_list


# if __name__ == '__main__':
#     simulate_channel()
