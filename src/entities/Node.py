import simpy


class Node:
    # def __init__(self, env, node_parameters, rebalancing_parameters, demand_estimates, verbose):
    def __init__(self, env, node_parameters, rebalancing_parameters, verbose):
        self.env = env
        self.local_balances = {"L": node_parameters["initial_balance_L"], "R": node_parameters["initial_balance_R"]}
        self.remote_balances = {"L": node_parameters["capacity_L"] - node_parameters["initial_balance_L"], "R": node_parameters["capacity_R"] - node_parameters["initial_balance_R"]}
        self.capacities = {"L": node_parameters["capacity_L"], "R": node_parameters["capacity_R"]}
        self.fees = [node_parameters["base_fee"], node_parameters["proportional_fee"]]
        self.on_chain_budget = node_parameters["on_chain_budget"]
        self.swap_IN_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.swap_OUT_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.rebalancing_parameters = rebalancing_parameters
        self.verbose = verbose
        self.latest_transactions_buffer_size = 20
        self.latest_transactions_times = {"L": [], "R": []}
        self.latest_transactions_amounts = {"L": [], "R": []}
        self.demand_estimates = {"L": 0.0, "R": 0.0}
        self.net_demands = {"L": 0.0, "R": 0.0}

        self.node_processor = simpy.Resource(env, capacity=1)
        self.rebalancing_locks = {"L": simpy.Resource(env, capacity=1), "R": simpy.Resource(env, capacity=1)}     # max 1 rebalancing operation active at a time
        # self.rebalance_requests = {"L": self.env.event(), "R": self.env.event()}
        self.time_to_check = self.env.event()

        self.balance_history_times = []
        self.balance_history_values = {"L": [], "R": []}
        self.total_fortune_including_pending_swaps_times = []
        self.total_fortune_including_pending_swaps_values = []
        self.total_fortune_including_pending_swaps_minus_losses_values = []
        self.fee_losses_over_time = []
        self.cumulative_fee_losses = 0.0
        self.rebalancing_fees_since_last_transaction = 0.0
        self.rebalancing_fees_over_time = []
        self.cumulative_rebalancing_fees = 0.0
        self.rebalancing_history_start_times = []
        self.rebalancing_history_end_times = []
        self.rebalancing_history_types = []
        self.rebalancing_history_amounts = []
        self.rebalancing_history_results = []


    def calculate_relay_fees(self, transaction_amount):
        if transaction_amount <= 0:
            relay_fees = 0
        else:
            relay_fees = self.fees[0] + transaction_amount*self.fees[1]
        return relay_fees

    def calculate_swap_in_fees(self, swap_in_amount):
        # swap_in_amount is the net amount that is moving IN THE CHANNEL: it DOES NOT include the fees
        if swap_in_amount <= 0:
            swap_in_fees = 0
        else:
            swap_in_fees = swap_in_amount * self.rebalancing_parameters["server_swap_fee"] + self.rebalancing_parameters["miner_fee"]
        return swap_in_fees

    def calculate_swap_out_fees(self, swap_out_amount):
        # swap_out_amount is the net amount that is moving IN THE CHANNEL: it DOES include the fees
        if swap_out_amount <= 0:
            swap_out_fees = 0
        else:
            net_amount_to_be_added_on_chain = (swap_out_amount - self.rebalancing_parameters["miner_fee"]) / (1 + self.rebalancing_parameters["server_swap_fee"])
            swap_out_fees = net_amount_to_be_added_on_chain * self.rebalancing_parameters["server_swap_fee"] + self.rebalancing_parameters["miner_fee"]
        return swap_out_fees

    def execute_feasible_transaction(self, t):
        # Calling this function requires checking for transaction feasibility beforehand. The function itself does not perform any checks, and this could lead to negative balances if misused.

        self.local_balances[t.previous_node] += t.amount
        self.remote_balances[t.previous_node] -= t.amount
        self.local_balances[t.next_node] -= (t.amount - self.calculate_relay_fees(t.amount))
        self.remote_balances[t.next_node] += (t.amount - self.calculate_relay_fees(t.amount))
        self.balance_history_times.append(self.env.now)
        self.balance_history_values["L"].append(self.local_balances["L"])
        self.balance_history_values["R"].append(self.local_balances["R"])
        t.status = "SUCCEEDED"

        if self.verbose:
            print("Time {:.2f}: SUCCESS: Transaction {} processed.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def reject_transaction(self, t):
        t.status = "FAILED"
        if self.verbose:
            print("Time {:.2f}: FAILURE: Transaction {} rejected.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def process_transaction(self, t):
        # Update the memory of the latest transactions
        if len(self.latest_transactions_times[t.source]) >= self.latest_transactions_buffer_size:
            self.latest_transactions_times[t.source].pop(0)
        self.latest_transactions_times[t.source].append(t.time_of_arrival)
        if len(self.latest_transactions_amounts[t.source]) >= self.latest_transactions_buffer_size:
            self.latest_transactions_amounts[t.source].pop(0)
        self.latest_transactions_amounts[t.source].append(t.amount)

        # If feasible, execute; otherwise, reject
        if (t.amount >= self.calculate_relay_fees(t.amount)) and (t.amount <= self.remote_balances[t.previous_node]) and (t.amount - self.calculate_relay_fees(t.amount) <= self.local_balances[t.next_node]):
            self.execute_feasible_transaction(t)
            fee_losses_of_this_transaction = 0.0
        else:
            self.reject_transaction(t)
            fee_losses_of_this_transaction = self.calculate_relay_fees(t.amount)
        # t.cleared.succeed()
        self.time_to_check.succeed()
        self.time_to_check = self.env.event()

        # Update logs and metrics
        self.total_fortune_including_pending_swaps_times.append(self.env.now)
        total_fortune_including_pending_swaps = self.local_balances["L"] + self.local_balances["R"] + self.on_chain_budget + self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"] + self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]
        total_fortune_including_pending_swaps_minus_losses = total_fortune_including_pending_swaps - self.cumulative_fee_losses
        self.total_fortune_including_pending_swaps_values.append(total_fortune_including_pending_swaps)
        self.total_fortune_including_pending_swaps_minus_losses_values.append(total_fortune_including_pending_swaps_minus_losses)
        self.fee_losses_over_time.append(fee_losses_of_this_transaction)
        self.cumulative_fee_losses += fee_losses_of_this_transaction
        self.rebalancing_fees_over_time.append(self.rebalancing_fees_since_last_transaction)    # We register rebalancing fees only when the next transaction arrives, and not at the actual time the rebalancing starts.
        self.cumulative_rebalancing_fees += self.rebalancing_fees_since_last_transaction
        self.rebalancing_fees_since_last_transaction = 0.0


    def perform_rebalancing_if_needed(self, neighbor):
        if self.verbose:
            print("Time {:.2f}: SWAP check performed for channel N-{}.".format(self.env.now, neighbor))

        if self.rebalancing_locks[neighbor].count == 0:     # if no rebalancing in progress in the N-neighbor channel
            with self.rebalancing_locks[neighbor].request() as rebalance_request:  # Generate a request event
                yield rebalance_request

                if self.rebalancing_parameters["rebalancing_policy"] == "none":
                    pass

                elif self.rebalancing_parameters["rebalancing_policy"] == "autoloop":
                    midpoint = self.capacities[neighbor] * (self.rebalancing_parameters["lower_threshold"] + self.rebalancing_parameters["upper_threshold"]) / 2

                    if self.local_balances[neighbor] < self.rebalancing_parameters["lower_threshold"] * self.capacities[neighbor]:    # SWAP-IN
                        swap_amount = midpoint - self.local_balances[neighbor]
                        yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                    elif self.local_balances[neighbor] > self.rebalancing_parameters["upper_threshold"] * self.capacities[neighbor]:      # SWAP-OUT
                        swap_amount = self.local_balances[neighbor] - midpoint
                        yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                    else:
                        pass    # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))
                #
                # elif self.rebalancing_parameters["rebalancing_policy"] == "autoloop-infrequent":
                #     midpoint = self.capacities[neighbor] * (self.rebalancing_parameters["lower_threshold"] + self.rebalancing_parameters["upper_threshold"]) / 2
                #     other_neighbor = "R" if neighbor == "L" else "L"
                #     self.net_demands[neighbor] = self.demand_estimates[neighbor] - (self.demand_estimates[other_neighbor] - self.calculate_fees(self.demand_estimates[other_neighbor]))
                #
                #     if self.balances[neighbor] < self.rebalancing_parameters["T_conf"] * self.net_demands[neighbor]:    # SWAP-IN
                #         swap_amount = midpoint - self.balances[neighbor]
                #         yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                #     elif self.balances[neighbor] > self.rebalancing_parameters["T_conf"] * self.net_demands[neighbor]:      # SWAP-OUT
                #         swap_amount = self.balances[neighbor] - midpoint
                #         yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                #     else:
                #         pass    # no rebalancing needed
                #         if self.verbose:
                #             print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))

                elif self.rebalancing_parameters["rebalancing_policy"] == "loopmax":
                    other_neighbor = "R" if neighbor == "L" else "L"

                    for n in [neighbor, other_neighbor]:
                        self.demand_estimates[n] = sum(self.latest_transactions_amounts[n]) / (self.latest_transactions_times[n][-1] - self.latest_transactions_times[n][0])

                    self.net_demands[neighbor] = self.demand_estimates[neighbor] - (self.demand_estimates[other_neighbor] - self.calculate_relay_fees(self.demand_estimates[other_neighbor]))

                    if self.net_demands[neighbor] < 0:  # SWAP-IN
                        expected_time_to_depletion = self.local_balances[neighbor] / (- self.net_demands[neighbor])
                        if expected_time_to_depletion - self.rebalancing_parameters["check_interval"] < self.rebalancing_parameters["T_conf"]:
                            safety_margin_in_coins = - self.net_demands[neighbor] * self.rebalancing_parameters["safety_margins_in_minutes"][neighbor]
                            swap_amount = self.max_swap_in_amount(neighbor) - safety_margin_in_coins
                            # swap_amount = self.max_swap_in_amount(neighbor)
                            yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                        else:
                            pass
                            if self.verbose:
                                print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))
                    elif self.net_demands[neighbor] > 0:    # SWAP-OUT
                        expected_time_to_saturation = self.remote_balances[neighbor] / self.net_demands[neighbor]
                        if expected_time_to_saturation - self.rebalancing_parameters["check_interval"] < self.rebalancing_parameters["T_conf"]:
                            safety_margin_in_coins = self.net_demands[neighbor] * self.rebalancing_parameters["safety_margins_in_minutes"][neighbor]
                            swap_amount = self.local_balances[neighbor] - safety_margin_in_coins
                            yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                        elif self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))
                    else:
                        pass    # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))
        else:
            pass  # if rebalancing already in progress, do not check again if rebalancing is needed
            if self.verbose:
                print("Time {:.2f}: SWAP already in progress in channel N-{}.".format(self.env.now, neighbor))

    def max_swap_in_amount(self, neighbor):
        # return min(self.on_chain_budget * (1 - self.rebalancing_parameters["server_swap_fee"]) - self.rebalancing_parameters["miner_fee"], self.capacities[neighbor])
        return min(self.on_chain_budget * (1 - self.rebalancing_parameters["server_swap_fee"]) - self.rebalancing_parameters["miner_fee"], self.remote_balances[neighbor])

    def swap_in(self, neighbor, swap_amount, rebalance_request):
        # swap_amount is the net amount that is moving IN THE CHANNEL: it DOES NOT include the fees
        swap_in_fees = self.calculate_swap_in_fees(swap_amount)

        self.rebalancing_history_start_times.append(self.env.now)
        self.rebalancing_history_types.append(neighbor + "-IN")
        self.rebalancing_history_amounts.append(swap_amount)        #TODO: this amount representation is different than in the theoretical model. This doesn't affect the correctness though.
        if self.verbose:
            print("Time {:.2f}: SWAP-IN initiated in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))

        if swap_amount <= 0:
            if self.verbose:
                print("Time {:.2f}: SWAP-IN aborted due to violation of safety margin in channel N-{}.".format(self.env.now, neighbor))
            self.rebalancing_history_results.append("ABORTED")
            self.rebalancing_history_end_times.append(self.env.now)
        elif self.on_chain_budget < swap_amount + swap_in_fees:
            if self.verbose:
                print("Time {:.2f}: SWAP-IN aborted due to insufficient balance in channel N-{}.".format(self.env.now, neighbor, swap_amount))
            self.rebalancing_history_results.append("ABORTED")
            self.rebalancing_history_end_times.append(self.env.now)
        else:
            self.on_chain_budget -= (swap_amount + swap_in_fees)
            self.swap_IN_amounts_in_progress[neighbor] += swap_amount
            self.rebalancing_fees_since_last_transaction += swap_in_fees
            yield self.env.timeout(self.rebalancing_parameters["T_conf"])

            self.swap_IN_amounts_in_progress[neighbor] -= swap_amount
            if self.remote_balances[neighbor] < swap_amount:
                if self.verbose:
                    print("Time {:.2f}: SWAP-IN failed in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))
                self.on_chain_budget += (swap_amount + swap_in_fees)                # Assumes immediate cancelation of swap-in operation and immediate refund
                self.rebalancing_fees_since_last_transaction -= swap_in_fees        # Assumes immediate cancelation of swap-in operation and immediate refund
                self.rebalancing_history_results.append("FAILED")
                self.rebalancing_history_end_times.append(self.env.now)
            else:   # success
                self.local_balances[neighbor] += swap_amount
                self.remote_balances[neighbor] -= swap_amount

                self.rebalancing_history_results.append("SUCCEEDED")
                self.rebalancing_history_end_times.append(self.env.now)
                self.balance_history_times.append(self.env.now)
                self.balance_history_values["L"].append(self.local_balances["L"])
                self.balance_history_values["R"].append(self.local_balances["R"])
                if self.verbose:
                    print("Time {:.2f}: SWAP-IN completed in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))
                    print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))
                # self.rebalance_requests[neighbor].succeed()
                # self.rebalance_requests[neighbor] = self.env.event()
                # return neighbor + "-in"
        self.rebalancing_locks[neighbor].release(rebalance_request)

    def swap_out(self, neighbor, swap_amount, rebalance_request):
        # swap_out_amount is the net amount that is moving IN THE CHANNEL: it DOES include the fees
        swap_out_fees = self.calculate_swap_out_fees(swap_amount)

        self.rebalancing_history_start_times.append(self.env.now)
        self.rebalancing_history_types.append(neighbor + "-OUT")
        self.rebalancing_history_amounts.append(swap_amount)
        if self.verbose:
            print("Time {:.2f}: SWAP-OUT initiated in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))

        if swap_amount <= 0:    # only possible under the Loopmax policy
            if self.verbose:
                print("Time {:.2f}: SWAP-OUT aborted due to violation of safety margin in channel N-{}.".format(self.env.now, neighbor))
            self.rebalancing_history_results.append("ABORTED")
            self.rebalancing_history_end_times.append(self.env.now)
        elif (self.local_balances[neighbor] < swap_amount) or (swap_amount < swap_out_fees):  # check the swap-out constraints
            if self.verbose:
                print("Time {:.2f}: SWAP-OUT aborted due to insufficient balance in channel N-{}.".format(self.env.now, neighbor))
            self.rebalancing_history_results.append("ABORTED")
            self.rebalancing_history_end_times.append(self.env.now)
        else:   # success
            self.local_balances[neighbor] -= swap_amount
            # self.swap_OUT_amounts_in_progress[neighbor] += (swap_amount - swap_out_fees)
            self.swap_OUT_amounts_in_progress[neighbor] += swap_amount
            self.rebalancing_fees_since_last_transaction += swap_out_fees
            yield self.env.timeout(self.rebalancing_parameters["T_conf"])
            self.on_chain_budget += (swap_amount - swap_out_fees)
            self.remote_balances[neighbor] += swap_amount
            # self.swap_OUT_amounts_in_progress[neighbor] -= (swap_amount - swap_out_fees)
            self.swap_OUT_amounts_in_progress[neighbor] -= swap_amount

            self.rebalancing_history_results.append("SUCCEEDED")
            self.rebalancing_history_end_times.append(self.env.now)
            self.balance_history_times.append(self.env.now)
            self.balance_history_values["L"].append(self.local_balances["L"])
            self.balance_history_values["R"].append(self.local_balances["R"])
            if self.verbose:
                print("Time {:.2f}: SWAP-OUT completed in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))
                print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))
            # self.rebalance_requests[neighbor].succeed()
            # self.rebalance_requests[neighbor] = self.env.event()
            # return neighbor + "-out"
        self.rebalancing_locks[neighbor].release(rebalance_request)

    def run(self):
        while True:
            # yield self.rebalance_requests["L"] | self.rebalance_requests["R"]

            # For checking after clearing each transaction
            # yield self.time_to_check

            # For checking every some fixed time
            yield self.env.timeout(self.rebalancing_parameters["check_interval"])

            for neighbor in ["L", "R"]:
                self.env.process(self.perform_rebalancing_if_needed(neighbor))
                # yield self.env.process(self.perform_rebalancing_if_needed(neighbor))
            # yield self.env.process(self.perform_rebalancing_if_needed("L"))
