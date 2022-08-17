import datetime

import numpy as np
import simpy
import torch
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

from src.learning.pytorch_soft_actor_critic.replay_memory import ReplayMemory
from src.learning.pytorch_soft_actor_critic.sac import SAC
from src.utils.MDP_utils import *


class Node:
    # def __init__(self, env, node_parameters, rebalancing_parameters, demand_estimates, verbose):
    # def __init__(self, env, node_parameters, rebalancing_parameters, verbose):
    def __init__(self, env, node_parameters, rebalancing_parameters, verbose, verbose_also_print_transactions, filename):
        self.env = env
        self.local_balances = {"L": node_parameters["initial_balance_L"], "R": node_parameters["initial_balance_R"]}
        self.remote_balances = {"L": node_parameters["capacity_L"] - node_parameters["initial_balance_L"], "R": node_parameters["capacity_R"] - node_parameters["initial_balance_R"]}
        self.capacities = {"L": node_parameters["capacity_L"], "R": node_parameters["capacity_R"]}
        self.fees = [node_parameters["base_fee"], node_parameters["proportional_fee"]]
        self.on_chain_budget = node_parameters["on_chain_budget"]
        # self.target_max_on_chain_amount = 100 * max(self.capacities["L"], self.capacities["R"])
        self.initial_fortune = node_parameters["initial_balance_L"] + node_parameters["initial_balance_R"] + node_parameters["on_chain_budget"]
        self.node_is_accepting_transactions = True

        self.swap_IN_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.swap_OUT_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.rebalancing_parameters = rebalancing_parameters
        self.verbose = verbose
        self.verbose_also_print_transactions = verbose_also_print_transactions
        # self.latest_transactions_buffer_size = 1000000000000
        self.latest_transactions = []
        self.latest_two_T_check_transactions = []
        self.total_amount_that_arrived_L_to_R = 0
        self.total_amount_that_arrived_R_to_L = 0
        self.total_number_of_txs_that_arrived_L_to_R = 0
        self.total_number_of_txs_that_arrived_R_to_L = 0
        self.total_amount_that_succeeded_L_to_R = 0
        self.total_amount_that_succeeded_R_to_L = 0
        self.A_hat_net_LN = 0
        self.A_hat_net_NL = 0
        self.A_hat_net_NR = 0
        self.A_hat_net_RN = 0
        self.S_hat_LR = 0
        self.S_hat_RL = 0
        self.S_hat_net_NL = 0
        self.S_hat_net_LN = 0
        self.S_hat_net_NR = 0
        self.S_hat_net_RN = 0
        self.b_hat_LN_future_estimate = 0
        self.b_hat_RN_future_estimate = 0
        self.max_swap_in_amount_due_to_current_constraints = self.capacities
        self.max_swap_out_amount_due_to_current_constraints = self.capacities

        if self.rebalancing_parameters["rebalancing_policy"] == "RebEL":
            self.learning_parameters = LearningParameters()
            self.target_max_on_chain_amount = self.on_chain_budget * self.learning_parameters.on_chain_normalization_multiplier
            torch.manual_seed(self.learning_parameters.seed)
            # self.observation_space = spaces.Box(
            #         low=np.zeros(5),
            #         high=np.array(
            #             [self.capacities["L"], self.capacities["L"], self.capacities["R"], self.capacities["R"], np.Inf]),
            #         shape=(5,),
            #         dtype=float
            #     )
            self.observation_space = spaces.Box(
                    # low=np.zeros(5),
                    low=np.zeros(7),
                    # high=np.ones(5),
                    high=np.ones(7),
                    # shape=(5,),
                    shape=(7,),
                    dtype=float
                )
            # self.action_space = spaces.Box(
            #     low=np.array([- self.capacities["L"], - self.capacities["R"]]),
            #     high=np.array([self.capacities["L"], self.capacities["R"]]),
            #     shape=(2,),
            #     dtype=float,
            #     seed=self.learning_parameters.seed
            # )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                shape=(2,),
                dtype=float,
                seed=self.learning_parameters.seed
            )
            self.agent = SAC(self.observation_space.shape[0], self.action_space, self.learning_parameters)
            self.replay_memory = ReplayMemory(
                capacity=self.learning_parameters.replay_size,
                seed=self.learning_parameters.seed
            )
            self.update_count = 0
            self.reward_normalizer = self.initial_fortune / 1000
            self.writer = SummaryWriter('./outputs/runs/{}_{}_SAC_{}'.format(filename, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "autotune" if self.learning_parameters.automatic_entropy_tuning else ""))    # Tensorboard setup
            self.min_swap_threshold_as_percentage_of_capacity = self.learning_parameters.min_swap_threshold_as_percentage_of_capacity
            self.swap_failure_penalty_coefficient = self.learning_parameters.swap_failure_penalty_coefficient
            self.steps_in_current_episode = 0
            self.total_steps = 0
            self.episode_is_done = False
        else:
            self.learning_parameters = None
            self.action_space = None
            self.agent = None
            self.replay_memory = None

        self.node_processor = simpy.Resource(env, capacity=1)
        self.rebalancing_locks = {"L": simpy.Resource(env, capacity=1), "R": simpy.Resource(env, capacity=1)}     # max 1 rebalancing operation active at a time
        # self.rebalance_requests = {"L": self.env.event(), "R": self.env.event()}
        self.time_to_check = self.env.event()
        self.min_swap_out_amount = self.rebalancing_parameters["miner_fee"] / (1 - self.rebalancing_parameters["server_swap_fee"])
        self.swap_in_successful_in_channel = []
        self.swap_out_successful_in_channel = []
        self.fee_losses_since_last_check_time = 0
        # self.fee_losses_since_two_check_times_ago = 0
        self.fee_profits_since_last_check_time = 0
        # self.fee_profits_since_two_check_times_ago = 0
        self.check_time_index = 0

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
            net_amount_to_be_added_on_chain = self.phi_inverse(swap_out_amount)
            swap_out_fees = net_amount_to_be_added_on_chain * self.rebalancing_parameters["server_swap_fee"] + self.rebalancing_parameters["miner_fee"]
        return swap_out_fees

    def phi_inverse(self, gross_amount):
        net_amount = (gross_amount - self.rebalancing_parameters["miner_fee"]) / (1 + self.rebalancing_parameters["server_swap_fee"])
        return net_amount

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

        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: SUCCESS: Transaction {} processed.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def reject_transaction(self, t):
        t.status = "FAILED"
        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: FAILURE: Transaction {} rejected.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def process_transaction(self, t):
        # # Update the memory of the latest transactions
        # if len(self.latest_transactions) >= self.latest_transactions_buffer_size:
        #     self.latest_transactions.pop(0)
        # self.latest_transactions.append(t)

        # Buffer of transactions of last two check times used for Reward 13
        # self.latest_two_T_check_transactions.append(t)
        # # Prune old ones
        # i = 0
        # while self.latest_two_T_check_transactions[i].time_of_arrival < self.env.now - 2*self.rebalancing_parameters["check_interval"]:
        #     i += 1
        # self.latest_two_T_check_transactions[0:i] = []

        tx_relay_fees = self.calculate_relay_fees(t.amount)

        # If feasible, execute; otherwise, reject
        if (t.amount >= tx_relay_fees) and (t.amount <= self.remote_balances[t.previous_node]) and (t.amount - tx_relay_fees <= self.local_balances[t.next_node]) and self.node_is_accepting_transactions:
            self.execute_feasible_transaction(t)
            fee_losses_of_this_transaction = 0.0
            self.total_amount_that_succeeded_L_to_R += t.amount if t.source == "L" and t.destination == "R" else 0
            self.total_amount_that_succeeded_R_to_L += t.amount if t.source == "R" and t.destination == "L" else 0
        else:
            self.reject_transaction(t)
            fee_losses_of_this_transaction = tx_relay_fees
        # t.cleared.succeed()
        self.time_to_check.succeed()
        self.time_to_check = self.env.event()

        # Update logs and metrics
        self.total_amount_that_arrived_L_to_R += t.amount if t.source == "L" and t.destination == "R" else 0
        self.total_amount_that_arrived_R_to_L += t.amount if t.source == "R" and t.destination == "L" else 0
        self.total_number_of_txs_that_arrived_L_to_R += 1 if t.source == "L" and t.destination == "R" else 0
        self.total_number_of_txs_that_arrived_R_to_L += 1 if t.source == "R" and t.destination == "L" else 0
        self.total_fortune_including_pending_swaps_times.append(self.env.now)
        total_fortune_including_pending_swaps = self.local_balances["L"] + self.local_balances["R"] + self.on_chain_budget + self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"] + self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]
        total_fortune_including_pending_swaps_minus_losses = total_fortune_including_pending_swaps - self.cumulative_fee_losses
        self.total_fortune_including_pending_swaps_values.append(total_fortune_including_pending_swaps)
        self.total_fortune_including_pending_swaps_minus_losses_values.append(total_fortune_including_pending_swaps_minus_losses)
        self.fee_losses_over_time.append(fee_losses_of_this_transaction)
        self.fee_losses_since_last_check_time += fee_losses_of_this_transaction
        self.fee_profits_since_last_check_time += tx_relay_fees - fee_losses_of_this_transaction
        self.cumulative_fee_losses += fee_losses_of_this_transaction
        self.rebalancing_fees_over_time.append(self.rebalancing_fees_since_last_transaction)    # We register rebalancing fees only when the next transaction arrives, and not at the actual time the rebalancing starts.
        self.cumulative_rebalancing_fees += self.rebalancing_fees_since_last_transaction
        self.rebalancing_fees_since_last_transaction = 0.0


    # def perform_rebalancing_if_needed(self, neighbor):
    def perform_rebalancing_if_needed(self):
        if self.rebalancing_parameters["rebalancing_policy"] == "None":
            return

        self.update_estimates()
        # calculate_estimates_formula_3(self)

        if self.rebalancing_parameters["rebalancing_policy"] in ["Autoloop", "Loopmax"]:
            for neighbor in ["L", "R"]:
                self.env.process(self.perform_rebalancing_if_needed_in_single_channel(neighbor=neighbor))

        elif self.rebalancing_parameters["rebalancing_policy"] == "RebEL":

            if self.episode_is_done:  # if episode is over, then perform reset to a balanced state
                if self.verbose:
                    print("Time {:.2f}: Resetting channels to balanced state.".format(self.env.now))
                    print("Time {:.2f}: Current balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

                self.node_is_accepting_transactions = False
                for neighbor in ["L", "R"]:
                    self.env.process(self.perform_rebalancing_if_needed_in_single_channel(neighbor=neighbor, reset=True))

                self.steps_in_current_episode = 0
                self.episode_is_done = False    # Did not use this variable as a flag inside the two parallel calls of the perform_rebalancing_if_needed_in_single_channel() function because of concurrency

            else:   # perform regular action as given by the learned policy

                self.node_is_accepting_transactions = True

                if self.verbose:
                    print("Time {:.2f}: SWAP check performed for all channels.".format(self.env.now))
                    print("Time {:.2f}: Current balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

                # if (self.rebalancing_locks["L"].count == 0) and (self.rebalancing_locks["R"].count == 0):  # if no rebalancing is in progress in any channel
                with self.rebalancing_locks["L"].request() as rebalance_request_L, self.rebalancing_locks["R"].request() as rebalance_request_R:  # Generate a request event for all channel locks
                    yield rebalance_request_L & rebalance_request_R

                    # Update parameters of all the networks if needed
                    if (len(self.replay_memory) > self.learning_parameters.batch_size) and (self.steps_in_current_episode > 0) and (self.total_steps % self.learning_parameters.nn_update_interval == 0):
                        for i in range(self.learning_parameters.updates_per_step):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(
                                memory=self.replay_memory,
                                batch_size=self.learning_parameters.batch_size,
                                updates=self.update_count
                            )
                            self.writer.add_scalar('loss/critic_1', critic_1_loss, self.update_count)
                            self.writer.add_scalar('loss/critic_2', critic_2_loss, self.update_count)
                            self.writer.add_scalar('loss/policy', policy_loss, self.update_count)
                            self.writer.add_scalar('loss/entropy_loss', ent_loss, self.update_count)
                            self.writer.add_scalar('entropy_temperature/alpha', alpha, self.update_count)
                            self.update_count += 1

                    # Perform environment step

                    # state = [
                    #     self.remote_balances["L"],
                    #     self.local_balances["L"],
                    #     self.local_balances["R"],
                    #     self.remote_balances["R"],
                    #     self.on_chain_budget,
                    #     # self.swap_IN_amounts_in_progress["L"],
                    #     # self.swap_OUT_amounts_in_progress["L"],
                    #     # self.swap_IN_amounts_in_progress["R"],
                    #     # self.swap_OUT_amounts_in_progress["R"]
                    # ]
                    state = [
                        self.remote_balances["L"] / self.capacities["L"],
                        self.local_balances["L"] / self.capacities["L"],
                        self.local_balances["R"] / self.capacities["R"],
                        self.remote_balances["R"] / self.capacities["R"],
                        self.on_chain_budget / self.target_max_on_chain_amount,
                        self.b_hat_LN_future_estimate / self.capacities["L"],
                        self.b_hat_RN_future_estimate / self.capacities["R"]
                    ]

                    state_absolute = [
                        state[0] * self.capacities["L"],
                        state[1] * self.capacities["L"],
                        state[2] * self.capacities["R"],
                        state[3] * self.capacities["R"],
                        state[4] * self.target_max_on_chain_amount,
                        state[5] * self.capacities["L"],
                        state[6] * self.capacities["R"]
                    ]

                    self.max_swap_in_amount_due_to_current_constraints = {
                        "L": min(state[5] * self.capacities["L"], self.phi_inverse(state[4] * self.target_max_on_chain_amount), self.capacities["L"]),
                        "R": min(state[6] * self.capacities["R"], self.phi_inverse(state[4] * self.target_max_on_chain_amount), self.capacities["R"])
                    }
                    self.max_swap_out_amount_due_to_current_constraints = {
                        "L": state[1] * self.capacities["L"],
                        "R": state[2] * self.capacities["R"]
                    }

                    # constraint_is_respected = False
                    # while not constraint_is_respected:  # ensure raw_action does not contain swap-ins for both channels
                    #     # raw_action = self.agent.select_action(state)  # raw_action = (r_L, r_R), possibly off-constraint bounds
                    #     if self.check_time_index < self.learning_parameters.start_steps:
                    #         raw_action = self.action_space.sample()  # Sample random action
                    #     else:
                    #         raw_action = self.agent.select_action(state)  # Sample action from policy
                    #     constraint_is_respected = rebalancing_amounts_not_both_positive(raw_action)
                    if self.check_time_index < self.learning_parameters.start_steps:
                        raw_action = self.action_space.sample()  # Sample random action
                    else:
                        raw_action = self.agent.select_action(state)  # Sample action from policy
                        # raw_action = self.agent.select_action_within_constraints(state)  # Sample action from policy, making sure the acton respects the current constraints

                    raw_action_absolute = [
                        raw_action[0] * self.max_swap_out_amount_due_to_current_constraints["L"] if raw_action[0] < 0 else raw_action[0] * self.max_swap_in_amount_due_to_current_constraints["L"],
                        raw_action[1] * self.max_swap_out_amount_due_to_current_constraints["R"] if raw_action[1] < 0 else raw_action[1] * self.max_swap_in_amount_due_to_current_constraints["R"]
                    ]

                    # processed_action = process_action_to_respect_constraints(   # processed_action = (r_L, r_R), inside constraint bounds or zero
                    #         raw_action=raw_action,
                    #         state=state,
                    #         N=self,
                    #     )
                    # processed_action = raw_action       # TODO: check if needed
                    # processed_action = process_action_to_be_more_than_min_rebalancing_percentage_v1(
                    #     raw_action=raw_action,
                    #     N=self,
                    # )
                    processed_action = process_action_to_be_more_than_min_rebalancing_percentage_v2(
                        raw_action=raw_action,
                        N=self,
                    )


                    # expanded_action = expand_action(processed_action)   # expanded action = (r_L_in, r_L_out, r_R_in, r_R_out)
                    # [r_L_in, r_L_out, r_R_in, r_R_out] = expanded_action

                    # if [r_L_in, r_L_out, r_R_in, r_R_out] == [0.0, 0.0, 0.0, 0.0]:
                    #     pass
                    #     if self.verbose:
                    #         print("Time {:.2f}: SWAP not needed in any channel of node N.".format(self.env.now))
                    # # elif
                    # else:   # execute non-zero swaps
                    #     if (r_L_in > 0.0) and (r_R_out > 0.0):
                    #         yield self.env.process(self.swap_in(
                    #             neighbor="L",
                    #             swap_amount=r_L_in,
                    #             rebalance_request=rebalance_request_L)
                    #         ) & self.env.process(self.swap_out(
                    #             neighbor="R",
                    #             swap_amount=r_R_out,
                    #             rebalance_request=rebalance_request_R)
                    #         )
                    #     elif (r_L_in > 0.0) and (r_R_out == 0.0):
                    #         yield self.env.process(self.swap_in(
                    #             neighbor="L",
                    #             swap_amount=r_L_in,
                    #             rebalance_request=rebalance_request_L)
                    #         )
                    #     elif (r_L_in == 0.0) and (r_R_out > 0.0):
                    #         yield self.env.process(self.swap_out(
                    #             neighbor="R",
                    #             swap_amount=r_R_out,
                    #             rebalance_request=rebalance_request_R)
                    #         )
                    #     elif (r_L_out > 0.0) and (r_R_in > 0.0):
                    #         yield self.env.process(self.swap_out(
                    #             neighbor="L",
                    #             swap_amount=r_L_out,
                    #             rebalance_request=rebalance_request_L)
                    #         ) & self.env.process(self.swap_in(
                    #             neighbor="R",
                    #             swap_amount=r_R_in,
                    #             rebalance_request=rebalance_request_R)
                    #         )
                    #     elif (r_L_out > 0.0) and (r_R_in == 0.0):
                    #         yield self.env.process(self.swap_out(
                    #             neighbor="L",
                    #             swap_amount=r_L_out,
                    #             rebalance_request=rebalance_request_L)
                    #         )
                    #     elif (r_L_out == 0) and (r_R_in > 0):
                    #         yield self.env.process(self.swap_in(
                    #             neighbor="R",
                    #             swap_amount=r_R_in,
                    #             rebalance_request=rebalance_request_R)
                    #         )
                    #     elif (r_L_out > 0.0) and (r_R_out > 0.0):
                    #         yield self.env.process(self.swap_out(
                    #             neighbor="L",
                    #             swap_amount=r_L_out,
                    #             rebalance_request=rebalance_request_L)
                    #         ) & self.env.process(self.swap_out(
                    #             neighbor="R",
                    #             swap_amount=r_R_out,
                    #             rebalance_request=rebalance_request_R)
                    #         )
                    #
                    #     else:
                    #         print("Unreachable point reached: invalid swap amount combination.")
                    #         exit(1)

                        # # -----------------------------
                        # if r_L_in > 0:
                        #     yield self.env.process(self.swap_in(
                        #         neighbor="L",
                        #         swap_amount=r_L_in,
                        #         rebalance_request=rebalance_request_L)
                        #     )
                        # else:   # if r_L_in == 0
                        #     pass
                        #
                        # if r_L_out > 0:
                        #     yield self.env.process(self.swap_out(
                        #         neighbor="L",
                        #         swap_amount=r_L_out,
                        #         rebalance_request=rebalance_request_L)
                        #     )
                        # else:   # if r_L_out == 0
                        #     pass
                        #
                        # if r_R_in > 0:
                        #     yield self.env.process(self.swap_in(
                        #         neighbor="R",
                        #         swap_amount=r_R_in,
                        #         rebalance_request=rebalance_request_R)
                        #     )
                        # else:   # if r_R_in == 0
                        #     pass
                        #
                        # if r_R_out > 0:
                        #     yield self.env.process(self.swap_out(
                        #         neighbor="R",
                        #         swap_amount=r_R_out,
                        #         rebalance_request=rebalance_request_R)
                        #     )
                        # else:   # if r_R_out == 0
                        #     pass

                    # next_state, reward, done, _ = rlenv.step(action)

                    [r_L, r_R] = processed_action
                    processed_action_absolute = {
                        "L": r_L * (self.max_swap_out_amount_due_to_current_constraints["L"] if r_L < 0 else self.max_swap_in_amount_due_to_current_constraints["L"]),
                        "R": r_R * (self.max_swap_out_amount_due_to_current_constraints["R"] if r_R < 0 else self.max_swap_in_amount_due_to_current_constraints["R"]),
                    }

                    # execute non-zero swaps
                    if r_L > 0.0:
                        if r_R > 0.0:
                            # print("Unreachable point reached: swap-ins in both channels is not a valid swap combination.")
                            # exit(1)
                            yield self.env.process(self.swap_in(
                                neighbor="L",
                                # swap_amount=r_L * self.capacities["L"],
                                swap_amount=processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            ) & self.env.process(self.swap_in(
                                neighbor="R",
                                swap_amount=processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )
                        elif r_R == 0:
                            yield self.env.process(self.swap_in(
                                neighbor="L",
                                # swap_amount=r_L,
                                # swap_amount=r_L * self.capacities["L"],
                                swap_amount=processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            )
                            if self.verbose:
                                print("Time {:.2f}: No SWAP performed in channel N-R.". format(self.env.now))
                        else:   # if r_R < 0.0
                            yield self.env.process(self.swap_in(
                                neighbor="L",
                                # swap_amount=r_L,
                                # swap_amount=r_L * self.capacities["L"],
                                swap_amount=processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            ) & self.env.process(self.swap_out(
                                neighbor="R",
                                # swap_amount=-r_R,
                                # swap_amount=-r_R * self.capacities["R"],
                                swap_amount=-processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )

                    elif r_L == 0.0:
                        if self.verbose:
                            print("Time {:.2f}: No SWAP performed in channel N-L.".format(self.env.now))
                        if r_R > 0.0:
                            yield self.env.process(self.swap_in(
                                neighbor="R",
                                # swap_amount=r_R,
                                # swap_amount=r_R * self.capacities["R"],
                                swap_amount=processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )
                        elif r_R == 0:
                            pass
                            if self.verbose:
                                print("Time {:.2f}: No SWAP performed in channel N-R.".format(self.env.now))
                            yield self.env.timeout(self.rebalancing_parameters["T_conf"])   # wait for T_conf so that the correct reward is accumulated from dropped transactions
                        else:   # if r_R < 0.0
                            yield self.env.process(self.swap_out(
                                neighbor="R",
                                # swap_amount=-r_R,
                                # swap_amount=-r_R * self.capacities["R"],
                                swap_amount=-processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )
                    else:   # if r_L < 0.0
                        if r_R > 0.0:
                            yield self.env.process(self.swap_out(
                                neighbor="L",
                                # swap_amount=-r_L,
                                # swap_amount=-r_L * self.capacities["L"],
                                swap_amount=-processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            ) & self.env.process(self.swap_in(
                                neighbor="R",
                                # swap_amount=r_R,
                                # swap_amount=r_R * self.capacities["R"],
                                swap_amount=processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )
                        elif r_R == 0.0:
                            yield self.env.process(self.swap_out(
                                neighbor="L",
                                # swap_amount=-r_L,
                                # swap_amount=-r_L * self.capacities["L"],
                                swap_amount=-processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            )
                            if self.verbose:
                                print("Time {:.2f}: No SWAP performed in channel N-R.".format(self.env.now))
                        else:  # if r_R < 0.0
                            yield self.env.process(self.swap_out(
                                neighbor="L",
                                # swap_amount=-r_L,
                                # swap_amount=-r_L * self.capacities["L"],
                                swap_amount=-processed_action_absolute["L"],
                                rebalance_request=rebalance_request_L)
                            ) & self.env.process(self.swap_out(
                                neighbor="R",
                                # swap_amount=-r_R,
                                # swap_amount=-r_R * self.capacities["R"],
                                swap_amount=-processed_action_absolute["R"],
                                rebalance_request=rebalance_request_R)
                            )

                    self.update_estimates()
                    # calculate_estimates_formula_3(self)

                    # next_state = [
                    #     self.remote_balances["L"],
                    #     self.local_balances["L"],
                    #     self.local_balances["R"],
                    #     self.remote_balances["R"],
                    #     self.on_chain_budget,
                    # ]
                    next_state = [
                        self.remote_balances["L"] / self.capacities["L"],
                        self.local_balances["L"] / self.capacities["L"],
                        self.local_balances["R"] / self.capacities["R"],
                        self.remote_balances["R"] / self.capacities["R"],
                        self.on_chain_budget / self.target_max_on_chain_amount,
                        self.b_hat_LN_future_estimate / self.capacities["L"],
                        self.b_hat_RN_future_estimate / self.capacities["R"]
                    ]

                    next_state_absolute = [
                        next_state[0] * self.capacities["L"],
                        next_state[1] * self.capacities["L"],
                        next_state[2] * self.capacities["R"],
                        next_state[3] * self.capacities["R"],
                        next_state[4] * self.target_max_on_chain_amount,
                        next_state[5] * self.capacities["L"],
                        next_state[6] * self.capacities["R"]
                    ]

                    # Reward 1
                    # reward = - ( self.fee_losses_since_last_check_time
                    #     +
                    #     (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else 0) +
                    #     self.calculate_swap_out_fees(r_L_out) +
                    #     (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else 0) +
                    #     self.calculate_swap_out_fees(r_R_out)
                    # )
                    [r_L_in, r_L_out, r_R_in, r_R_out] = expand_action(processed_action)   # expanded action = (r_L_in, r_L_out, r_R_in, r_R_out)
                    # Reward 2
                    # reward = - (self.fee_losses_since_last_check_time
                    #             +
                    #             # (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else 0) +
                    #             (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in)) +
                    #             # self.calculate_swap_in_fees(r_L_in) +
                    #             # self.calculate_swap_out_fees(r_L_out) +
                    #             (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out)) +
                    #             # (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else 0) +
                    #             (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in)) +
                    #             # self.calculate_swap_in_fees(r_R_in) +
                    #             # self.calculate_swap_out_fees(r_R_out)
                    #             (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                    #             )
                    # TODO: save rewards and respective times

                    # Reward 3
                    # reward = - (self.fee_losses_since_last_check_time / (max(self.capacities["L"], self.capacities["R"])) +
                    #             (self.calculate_swap_in_fees(r_L_in) / self.capacities["L"] if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in) / self.capacities["L"]) +
                    #             (self.calculate_swap_out_fees(r_L_out) / self.capacities["L"] if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out) / self.capacities["L"]) +
                    #             (self.calculate_swap_in_fees(r_R_in) / self.capacities["R"] if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in) / self.capacities["R"]) +
                    #             (self.calculate_swap_out_fees(r_R_out) / self.capacities["R"] if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out) / self.capacities["R"])
                    #             )

                    # Reward 4
                    # reward = - (self.fee_losses_since_last_check_time +
                    #             (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in)) +
                    #             (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out)) +
                    #             (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in)) +
                    #             (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                    #             ) / (max(self.capacities["L"], self.capacities["R"]))

                    total_fortune_before = state[1] * self.capacities["L"] + state[2] * self.capacities["R"] + state[4] * self.target_max_on_chain_amount
                    total_fortune_after = next_state[1] * self.capacities["L"] + next_state[2] * self.capacities["R"] + next_state[4] * self.target_max_on_chain_amount

                    # Reward 5
                    # reward = (total_fortune_after - total_fortune_before) / self.reward_normalizer

                    # Counting A
                    # number_of_swaps_chosen_in_the_wrong_direction = \
                    #     float((self.A_hat_net_NL > 0) and (raw_action[0] > 0)) + \
                    #     float((self.A_hat_net_NL < 0) and (raw_action[0] < 0)) + \
                    #     float((self.A_hat_net_NR > 0) and (raw_action[1] > 0)) + \
                    #     float((self.A_hat_net_NR < 0) and (raw_action[1] < 0))
                    # number_of_swaps_chosen_in_the_correct_direction = \
                    #     float((self.A_hat_net_NL < 0) and (raw_action[0] > 0)) + \
                    #     float((self.A_hat_net_NL > 0) and (raw_action[0] < 0)) + \
                    #     float((self.A_hat_net_NR < 0) and (raw_action[1] > 0)) + \
                    #     float((self.A_hat_net_NR > 0) and (raw_action[1] < 0))

                    # Counting B
                    # number_of_swaps_chosen_in_the_wrong_direction = \
                    #     float((self.S_hat_net_NL > 0) and (raw_action[0] > 0)) + \
                    #     float((self.S_hat_net_NL < 0) and (raw_action[0] < 0)) + \
                    #     float((self.S_hat_net_NR > 0) and (raw_action[1] > 0)) + \
                    #     float((self.S_hat_net_NR < 0) and (raw_action[1] < 0))
                    # number_of_swaps_chosen_in_the_correct_direction = \
                    #     float((self.S_hat_net_NL < 0) and (raw_action[0] > 0)) + \
                    #     float((self.S_hat_net_NL > 0) and (raw_action[0] < 0)) + \
                    #     float((self.S_hat_net_NR < 0) and (raw_action[1] > 0)) + \
                    #     float((self.S_hat_net_NR > 0) and (raw_action[1] < 0))

                    # Counting C
                    number_of_swaps_chosen_in_the_wrong_direction = \
                        float((self.S_hat_net_NL > 0) and (raw_action[0] > 0) and (processed_action_absolute["L"] > 0.0)) + \
                        float((self.S_hat_net_NL < 0) and (raw_action[0] < 0) and (processed_action_absolute["L"] > 0.0)) + \
                        float((self.S_hat_net_NR > 0) and (raw_action[1] > 0) and (processed_action_absolute["R"] > 0.0)) + \
                        float((self.S_hat_net_NR < 0) and (raw_action[1] < 0) and (processed_action_absolute["R"] > 0.0))
                    number_of_swaps_chosen_in_the_correct_direction = \
                        float((self.S_hat_net_NL < 0) and ((raw_action[0] > 0) or ((raw_action[0] < 0) and (processed_action_absolute["L"] == 0.0)))) + \
                        float((self.S_hat_net_NL > 0) and ((raw_action[0] < 0) or ((raw_action[0] > 0) and (processed_action_absolute["L"] == 0.0)))) + \
                        float((self.S_hat_net_NR < 0) and ((raw_action[1] > 0) or ((raw_action[0] < 0) and (processed_action_absolute["R"] == 0.0)))) + \
                        float((self.S_hat_net_NR > 0) and ((raw_action[1] < 0) or ((raw_action[0] > 0) and (processed_action_absolute["R"] == 0.0))))


                    number_of_zero_swaps = float(processed_action_absolute["L"] == 0.0) + float(processed_action_absolute["R"] == 0.0)

                    # Reward 6
                    # reward = (total_fortune_after - total_fortune_before
                    #           - self.fee_losses_since_last_check_time
                    #           + self.fee_profits_since_last_check_time * (0 if self.total_steps < 120 else 3)
                    #           + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction * (1 if self.total_steps < 120 else 1/5)
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction * (1 if self.total_steps < 120 else 1/5)
                    #           ) #/ (self.reward_normalizer)

                    # reward = (total_fortune_after - total_fortune_before
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction
                    #           ) / self.reward_normalizer

                    # Reward 7
                    reward = (total_fortune_after - total_fortune_before
                              - self.fee_losses_since_last_check_time
                                - (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in)) +
                                - (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out)) +
                                - (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in)) +
                                - (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                              + self.learning_parameters.bonus_for_zero_action * number_of_zero_swaps
                              + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction
                              - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction
                              ) #/ (self.reward_normalizer)

                    # Reward 8
                    # reward = (total_fortune_after - total_fortune_before
                    #           - self.fee_losses_since_last_check_time
                    #           + self.learning_parameters.zero_action_bonus * number_of_zero_swaps
                    #           + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction
                    #           - self.initial_fortune)# / (self.reward_normalizer)

                    # # Reward 9a
                    # # self.target_fortune = 100* self.initial_fortune
                    # self.target_fortune_increase = 10000
                    # reward = (total_fortune_after - total_fortune_before
                    #           - self.fee_losses_since_last_check_time
                    #           + self.learning_parameters.bonus_for_zero_action * number_of_zero_swaps
                    #           + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           #) / (self.reward_normalizer)
                    #           # - self.target_fortune) / self.initial_fortune
                    #           - self.target_fortune_increase)#*1000 / self.initial_fortune


                    # Reward 9b
                    # # self.target_fortune = 100* self.initial_fortune
                    # self.target_fortune_increase = 1000
                    # reward =          (self.fee_profits_since_last_check_time * 100
                    #           - (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else self.learning_parameters.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in))
                    #           - (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else self.learning_parameters.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out))
                    #           - (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else self.learning_parameters.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in))
                    #           - (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else self.learning_parameters.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                    #           - self.fee_losses_since_last_check_time
                    #           + self.learning_parameters.bonus_for_zero_action * number_of_zero_swaps
                    #           + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           #) / (self.reward_normalizer)
                    #           # - self.target_fortune) / self.initial_fortune
                    #           - self.target_fortune_increase
                    #           )#*100#0 / self.initial_fortune





                    # Reward 10 (rate)
                    # self.target_fortune_increase_rate = 1    #50 * self.initial_fortune - self.initial_fortune
                    # reward = (((total_fortune_after - total_fortune_before
                    #           - self.fee_losses_since_last_check_time
                    #           #) / (self.reward_normalizer)
                    #           ) / total_fortune_before - self.target_fortune_increase_rate)
                    #           + self.learning_parameters.bonus_for_zero_action * number_of_zero_swaps
                    #         + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #         - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #         )

                    # Reward 0
                    # reward = np.random.randint(-1000, 10)
                    # reward = np.random.randint(0, 1)
                    # reward = 0

                    # Reward 11
                    # reward = (self.fee_profits_since_last_check_time * min(self.capacities["L"], self.capacities["R"]) \
                    #             - self.fee_losses_since_last_check_time * min(self.capacities["L"], self.capacities["R"]) \
                    #             - (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else 0) * self.capacities["L"] - \
                    #             self.calculate_swap_out_fees(r_L_out) - \
                    #             (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else 0) * self.capacities["R"] - \
                    #             self.calculate_swap_out_fees(r_R_out)
                    #           ) / self.initial_fortune*100

                    # Reward 12
                    # reward = (self.fee_profits_since_last_check_time
                    #             - self.fee_losses_since_last_check_time
                    #             - (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in)) +
                    #             - (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out)) +
                    #             - (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in)) +
                    #             - (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                    #             - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction
                    #           ) / self.reward_normalizer


                    self.steps_in_current_episode += 1
                    self.episode_is_done = self.steps_in_current_episode >= self.learning_parameters.episode_duration
                    self.fee_losses_since_last_check_time = 0
                    self.fee_profits_since_last_check_time = 0
                    self.swap_in_successful_in_channel = []
                    self.swap_out_successful_in_channel = []

                    mask = float(not self.episode_is_done)


                    # # Reward 13 (reward of an action dependending on two check_intervals and not only one
                    # time_to_wait = 2*self.rebalancing_parameters["check_interval"] - self.rebalancing_parameters["T_conf"]-0.001
                    # yield self.env.timeout(time_to_wait)
                    #
                    # fees_of_last_two_check_interval_successes = sum(self.calculate_relay_fees(t.amount) for t in self.latest_two_T_check_transactions if (t.time_of_arrival >= self.env.now - rebalancing_start_time) and (t.status == 'SUCCEEDED'))
                    # fees_of_last_two_check_interval_losses = sum(self.calculate_relay_fees(t.amount) for t in self.latest_two_T_check_transactions if (t.time_of_arrival >= self.env.now - rebalancing_start_time) and (t.status == 'FAILED'))
                    #
                    # self.target_fortune_increase = 10000
                    # reward = ((self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else 0) +
                    #           self.calculate_swap_out_fees(r_L_out) +
                    #           (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else 0) +
                    #           self.calculate_swap_out_fees(r_R_out)
                    #           + fees_of_last_two_check_interval_successes
                    #           - fees_of_last_two_check_interval_losses
                    #           + self.learning_parameters.bonus_for_zero_action * number_of_zero_swaps
                    #           + self.learning_parameters.bonus_for_swap_in_correct_direction * number_of_swaps_chosen_in_the_correct_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           - self.learning_parameters.penalty_for_swap_in_wrong_direction * number_of_swaps_chosen_in_the_wrong_direction  # * (1 if self.total_steps < 120 else 1/5)
                    #           #) / (self.reward_normalizer)
                    #           # - self.target_fortune) / self.initial_fortune
                    #           - self.target_fortune_increase)#*1000 / self.initial_fortune

                    # self.fee_losses_since_two_check_times_ago = self.fee_losses_since_last_check_time
                    # self.fee_profits_since_two_check_times_ago = self.fee_profits_since_last_check_time

                    # self.replay_memory.push(state, processed_action, reward, next_state, mask)  # Append transition to memory
                    self.replay_memory.push(state, raw_action, reward, next_state, mask)  # Append transition to memory

                    if self.verbose:
                        # Print values as they are
                        # print_precision = "%.5f"
                        # state_printable = [print_precision % elem for elem in state]
                        # processed_action_printable = [print_precision % elem for elem in processed_action]
                        # raw_action_printable = [print_precision % elem for elem in raw_action]
                        # next_state_printable = [print_precision % elem for elem in next_state]
                        # print("Time {:.2f}: Tuple pushed to memory:\n\tstate = {}\n\tprocessed action = {} from raw action = {}\n\treward = {:.5f}\n\tnext state = {}".format(
                        #         self.env.now, state_printable, processed_action_printable, raw_action_printable, reward, next_state_printable
                        #     )
                        # )
                        # Print values converted to absolute amounts
                        processed_action_absolute_as_a_list = [processed_action_absolute[neighbor] for neighbor in ["L", "R"]]

                        print_precision = "%.2f"
                        state_printable = [print_precision % elem for elem in state_absolute]
                        processed_action_printable = [print_precision % elem for elem in processed_action_absolute_as_a_list]
                        raw_action_printable = [print_precision % elem for elem in raw_action]
                        raw_action_absolute_printable = [print_precision % elem for elem in raw_action_absolute]
                        next_state_printable = [print_precision % elem for elem in next_state_absolute]
                        print("Time {:.2f}: Tuple pushed to memory:\n\tstate absolute = {}\n\tprocessed action absolute = {} from raw action absolute = {} and raw action = {}\n\treward = {:.5f}\n\tnext state absolute = {}".format(
                                self.env.now, state_printable, processed_action_printable, raw_action_absolute_printable, raw_action_printable, reward, next_state_printable
                            )
                        )

            self.total_steps += 1

        else:
            print("{} is not a valid rebalancing policy.".format(self.rebalancing_parameters["rebalancing_policy"]))
            exit(1)


    def perform_rebalancing_if_needed_in_single_channel(self, neighbor, reset=False):
        if self.verbose and not reset:
            print("Time {:.2f}: SWAP check performed for channel N-{}.".format(self.env.now, neighbor))
            print("Time {:.2f}: Current balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(
                    self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"],
                    self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"],
                    self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

        if self.rebalancing_locks[neighbor].count == 0:  # if no rebalancing in progress in the N-neighbor channel
            with self.rebalancing_locks[neighbor].request() as rebalance_request:  # Generate a request event
                yield rebalance_request

                if self.rebalancing_parameters["rebalancing_policy"] == "Autoloop":
                    midpoint = self.capacities[neighbor] * (self.rebalancing_parameters["autoloop_lower_threshold"] + self.rebalancing_parameters["autoloop_upper_threshold"]) / 2

                    if self.local_balances[neighbor] < self.rebalancing_parameters["autoloop_lower_threshold"] * self.capacities[neighbor]:  # SWAP-IN
                        swap_amount = midpoint - self.local_balances[neighbor]
                        yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                    elif self.local_balances[neighbor] > self.rebalancing_parameters["autoloop_upper_threshold"] * self.capacities[neighbor]:  # SWAP-OUT
                        swap_amount = self.local_balances[neighbor] - midpoint
                        yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                    else:
                        pass  # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))

                elif self.rebalancing_parameters["rebalancing_policy"] == "Autoloop-infrequent":
                    midpoint = self.capacities[neighbor] * (self.rebalancing_parameters["autoloop_lower_threshold"] + self.rebalancing_parameters["autoloop_upper_threshold"]) / 2
                    net_rate_of_local_balance_change = {"L": self.A_hat_net_NL, "R": self.A_hat_net_NR}

                    if self.local_balances[neighbor] < self.rebalancing_parameters["T_conf"] * net_rate_of_local_balance_change[neighbor]:    # SWAP-IN
                        swap_amount = midpoint - self.local_balances[neighbor]
                        yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                    elif self.local_balances[neighbor] > self.rebalancing_parameters["T_conf"] * net_rate_of_local_balance_change[neighbor]:      # SWAP-OUT
                        swap_amount = self.local_balances[neighbor] - midpoint
                        yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                    else:
                        pass    # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.". format(self.env.now, neighbor))

                elif self.rebalancing_parameters["rebalancing_policy"] == "Loopmax":
                    net_rate_of_local_balance_change = {"L": self.A_hat_net_NL, "R": self.A_hat_net_NR}

                    if net_rate_of_local_balance_change[neighbor] < 0:  # SWAP-IN
                        expected_time_to_depletion = self.local_balances[neighbor] / (- net_rate_of_local_balance_change[neighbor])
                        if expected_time_to_depletion - self.rebalancing_parameters["check_interval"] < self.rebalancing_parameters["T_conf"]:
                            safety_margin_in_coins = - net_rate_of_local_balance_change[neighbor] * self.rebalancing_parameters["safety_margins_in_minutes"][neighbor]
                            # swap_amount = self.max_swap_in_amount(neighbor)
                            # swap_amount = self.max_swap_in_amount(neighbor) - safety_margin_in_coins
                            # swap_amount = min(self.max_swap_in_amount_allowed_by_on_chain_balance(), self.remote_balances[neighbor]) - safety_margin_in_coins
                            swap_amount = min(self.phi_inverse(self.on_chain_budget), self.remote_balances[neighbor]) - safety_margin_in_coins
                            yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                        else:
                            pass
                            if self.verbose:
                                print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))
                    elif net_rate_of_local_balance_change[neighbor] > 0:  # SWAP-OUT
                        expected_time_to_saturation = self.remote_balances[neighbor] / net_rate_of_local_balance_change[neighbor]
                        if expected_time_to_saturation - self.rebalancing_parameters["check_interval"] < self.rebalancing_parameters["T_conf"]:
                            safety_margin_in_coins = net_rate_of_local_balance_change[neighbor] * self.rebalancing_parameters["safety_margins_in_minutes"][neighbor]
                            swap_amount = self.local_balances[neighbor] - safety_margin_in_coins
                            yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                        elif self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))
                    else:
                        pass  # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))

                elif (self.rebalancing_parameters["rebalancing_policy"] == "RebEL") and (reset is True):

                    # Reset channel to "balanced" state
                    target_local_balance_absolute = self.learning_parameters.target_local_balance_fractions_at_state_reset[neighbor] * self.capacities[neighbor]

                    if self.local_balances[neighbor] < target_local_balance_absolute:  # SWAP-IN
                        swap_amount = target_local_balance_absolute - self.local_balances[neighbor]
                        yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                    elif self.local_balances[neighbor] > target_local_balance_absolute:  # SWAP-OUT
                        swap_amount = self.local_balances[neighbor] - target_local_balance_absolute
                        yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                    else:
                        pass  # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))

        else:
            pass  # if rebalancing already in progress, do not check again if rebalancing is needed
            if self.verbose:
                print("Time {:.2f}: SWAP already in progress in channel N-{}.".format(self.env.now, neighbor))

    def update_estimates(self):
        # Estimations of amounts added to the respective balance per minute (unit time)

        # # Estimate #1 based on finite buffer only and arrived amounts
        # timespan_of_transactions_in_buffer = (self.latest_transactions[-1].time_of_arrival - self.latest_transactions[0].time_of_arrival) if len(self.latest_transactions) > 1 else np.Inf
        # self.A_hat_net_NL = (
        #                         sum([t.amount for t in self.latest_transactions if t.source == "L"]) - sum(
        #                         [t.amount - self.calculate_relay_fees(t.amount) for t in self.latest_transactions if t.source == "R"])
        #                     ) / timespan_of_transactions_in_buffer
        # self.A_hat_net_LN = - self.A_hat_net_NL
        # self.A_hat_net_NR = (
        #                         sum([t.amount for t in self.latest_transactions if t.source == "R"]) - sum(
        #                         [t.amount - self.calculate_relay_fees(t.amount) for t in self.latest_transactions if t.source == "L"])
        #                     ) / timespan_of_transactions_in_buffer
        # self.A_hat_net_RN = - self.A_hat_net_NR
        #
        # if self.verbose:
        #     print("Time {:.2f}: A_hat_net_LN = {:.2f}, A_hat_net_NL = {:.2f}, A_hat_net_NR = {:.2f}, A_hat_net_RN = {:.2f}".format(self.env.now, self.A_hat_net_LN, self.A_hat_net_NL, self.A_hat_net_NR, self.A_hat_net_RN))

        # Estimate #2 based on all transactions that arrived since the simulation started
        self.A_hat_net_NL = (self.total_amount_that_arrived_L_to_R - (self.total_amount_that_arrived_R_to_L - self.calculate_relay_fees(self.total_amount_that_arrived_R_to_L))) / self.env.now
        self.A_hat_net_LN = - self.A_hat_net_NL
        self.A_hat_net_NR = (self.total_amount_that_arrived_R_to_L - (self.total_amount_that_arrived_L_to_R - self.calculate_relay_fees(self.total_amount_that_arrived_L_to_R))) / self.env.now
        self.A_hat_net_RN = - self.A_hat_net_NR

        self.S_hat_LR = self.total_amount_that_succeeded_L_to_R / self.env.now
        self.S_hat_RL = self.total_amount_that_succeeded_R_to_L / self.env.now
        self.S_hat_net_NL = (self.total_amount_that_succeeded_L_to_R - (self.total_amount_that_succeeded_R_to_L - self.calculate_relay_fees(self.total_amount_that_succeeded_R_to_L))) / self.env.now   # amount added to b_NL over time due to successful txs
        self.S_hat_net_LN = -self.S_hat_net_NL
        self.S_hat_net_NR = (self.total_amount_that_succeeded_R_to_L - (self.total_amount_that_succeeded_L_to_R - self.calculate_relay_fees(self.total_amount_that_succeeded_L_to_R))) / self.env.now
        self.S_hat_net_RN = - self.S_hat_net_NR

        if self.verbose:
            print("Time {:.2f}: A_hat_net_LN = {:.2f}, A_hat_net_NL = {:.2f}, A_hat_net_NR = {:.2f}, A_hat_net_RN = {:.2f}, S_hat_LR = {:.2f}, S_hat_RL = {:.2f}".format(self.env.now, self.A_hat_net_LN, self.A_hat_net_NL, self.A_hat_net_NR, self.A_hat_net_RN, self.S_hat_LR, self.S_hat_RL))

        # self.b_hat_LN_future_estimate = max(0, min(self.remote_balances["L"] + self.A_hat_net_LN * self.rebalancing_parameters["T_conf"], self.capacities["L"]))
        # self.b_hat_RN_future_estimate = max(0, min(self.remote_balances["R"] + self.A_hat_net_RN * self.rebalancing_parameters["T_conf"], self.capacities["R"]))
        #
        # if self.verbose:
        #     print("Time {:.2f}: Estimate 1:".format(self.env.now))
        #     print("Time {:.2f}: b_hat_LN_future_estimate = {:.2f}, b_hat_RN_future_estimate = {:.2f}".format(self.env.now, self.b_hat_LN_future_estimate, self.b_hat_RN_future_estimate))

        # Estimate #3 based on total successful amounts since the simulation started, NOT adjusted for channel depletion

        # self.b_hat_LN_future_estimate = max(0, min(self.remote_balances["L"] + ((self.S_hat_RL - self.calculate_relay_fees(self.S_hat_RL)) - self.S_hat_LR) * self.rebalancing_parameters["T_conf"], self.capacities["L"]))
        # self.b_hat_RN_future_estimate = max(0, min(self.remote_balances["R"] + ((self.S_hat_LR - self.calculate_relay_fees(self.S_hat_LR)) - self.S_hat_RL) * self.rebalancing_parameters["T_conf"], self.capacities["R"]))
        #
        # if self.verbose:
        #     print("Time {:.2f}: Estimate 2:".format(self.env.now))
        #     print("Time {:.2f}: b_hat_LN_future_estimate = {:.2f}, b_hat_RN_future_estimate = {:.2f}".format(self.env.now, self.b_hat_LN_future_estimate, self.b_hat_RN_future_estimate))

        # Estimate #4 based on total successful amounts since the simulation started, adjusted for channel depletion

        S_hat_LR_clipped = self.S_hat_LR * min(self.rebalancing_parameters["T_conf"], self.remote_balances["L"] / self.S_hat_LR, self.local_balances["R"] / self.S_hat_LR) if self.S_hat_LR > 0 else 0
        S_hat_RL_clipped = self.S_hat_RL * min(self.rebalancing_parameters["T_conf"], self.remote_balances["R"] / self.S_hat_RL, self.local_balances["L"] / self.S_hat_RL) if self.S_hat_RL > 0 else 0
        self.b_hat_LN_future_estimate = max(0, min(
            self.remote_balances["L"] - S_hat_LR_clipped + (S_hat_RL_clipped - self.calculate_relay_fees(S_hat_RL_clipped)),
            self.capacities["L"])
        )
        self.b_hat_RN_future_estimate = max(0, min(
            self.remote_balances["R"] - S_hat_RL_clipped + (S_hat_LR_clipped - self.calculate_relay_fees(S_hat_LR_clipped)),
            self.capacities["R"])
        )

        if self.verbose:
            print("Time {:.2f}: Estimate 4:".format(self.env.now))
            print("Time {:.2f}: b_hat_LN_future_estimate = {:.2f}, b_hat_RN_future_estimate = {:.2f}".format(self.env.now, self.b_hat_LN_future_estimate, self.b_hat_RN_future_estimate))

        # # Estimate #5 using mini-simulation of virtual transaction arrivals based on past successful amounts
        #
        # empirical_average_arriving_tx_size_LR = self.total_amount_that_arrived_L_to_R / self.total_number_of_txs_that_arrived_L_to_R
        # empirical_average_arriving_tx_size_RL = self.total_amount_that_arrived_R_to_L / self.total_number_of_txs_that_arrived_R_to_L
        # A_hat_LR = self.total_amount_that_arrived_L_to_R / self.env.now
        # A_hat_RL = self.total_amount_that_arrived_R_to_L / self.env.now
        # total_est_arriving_amount_LR_in_check_interval = A_hat_LR * self.rebalancing_parameters["check_interval"]
        # total_est_arriving_amount_RL_in_check_interval = A_hat_RL * self.rebalancing_parameters["check_interval"]
        #
        # split_A_hat_LR_number_of_txs = int(total_est_arriving_amount_LR_in_check_interval / empirical_average_arriving_tx_size_LR) + 1
        # leftover_amount_LR = total_est_arriving_amount_LR_in_check_interval - split_A_hat_LR_number_of_txs * empirical_average_arriving_tx_size_LR
        # split_A_hat_LR_amounts_of_each = list(np.ones(split_A_hat_LR_number_of_txs,) * empirical_average_arriving_tx_size_LR)
        # split_A_hat_LR_amounts_of_each.append(leftover_amount_LR)
        # split_A_hat_RL_number_of_txs = int(total_est_arriving_amount_RL_in_check_interval / empirical_average_arriving_tx_size_RL) + 1
        # leftover_amount_RL = total_est_arriving_amount_RL_in_check_interval - split_A_hat_RL_number_of_txs * empirical_average_arriving_tx_size_RL
        # split_A_hat_RL_amounts_of_each = list(np.ones(split_A_hat_RL_number_of_txs,) * empirical_average_arriving_tx_size_RL)
        # split_A_hat_RL_amounts_of_each.append(leftover_amount_RL)
        #
        # # split_S_hat_LR_number_of_txs = self.S_hat_LR / self.empirical_average_tx_size_LR
        # # split_S_hat_LR_amount_of_each = self.empirical_average_tx_size_LR
        # # split_S_hat_RL_number_of_txs = self.S_hat_RL / self.empirical_average_tx_size_RL
        # # split_S_hat_RL_amount_of_each = self.empirical_average_tx_size_RL
        #
        # est_local_balances = copy.deepcopy(self.local_balances)
        # est_remote_balances = copy.deepcopy(self.remote_balances)
        #
        # for i in range(max(split_A_hat_LR_number_of_txs, split_A_hat_RL_number_of_txs)):
        #     if i < split_A_hat_LR_number_of_txs:
        #         # Process an L-to-R virtual transaction
        #         tx_amount = split_A_hat_LR_amounts_of_each[i]
        #         tx_relay_fees = self.calculate_relay_fees(tx_amount)
        #         previous_node = "L"
        #         next_node = "R"
        #
        #         if (tx_amount >= tx_relay_fees) and (tx_amount <= est_remote_balances[previous_node]) and (tx_amount - tx_relay_fees <= est_local_balances[next_node]):
        #             # Execute virtual transaction
        #             est_local_balances[previous_node] += tx_amount
        #             est_remote_balances[previous_node] -= tx_amount
        #             est_local_balances[next_node] -= (tx_amount - tx_relay_fees)
        #             est_remote_balances[next_node] += (tx_amount - tx_relay_fees)
        #         else:
        #             # Reject virtual transaction
        #             pass
        #
        #     if i < split_A_hat_RL_number_of_txs:
        #         # Process an R-to-L virtual transaction
        #         tx_amount = split_A_hat_RL_amounts_of_each[i]
        #         tx_relay_fees = self.calculate_relay_fees(tx_amount)
        #         previous_node = "R"
        #         next_node = "L"
        #
        #         if (tx_amount >= tx_relay_fees) and (tx_amount <= est_remote_balances[previous_node]) and (tx_amount - tx_relay_fees <= est_local_balances[next_node]):
        #             est_local_balances[previous_node] += tx_amount
        #             est_remote_balances[previous_node] -= tx_amount
        #             est_local_balances[next_node] -= (tx_amount - tx_relay_fees)
        #             est_remote_balances[next_node] += (tx_amount - tx_relay_fees)
        #         else:
        #             # Reject virtual transaction
        #             pass
        #
        # self.b_hat_LN_future_estimate = est_remote_balances["L"]
        # self.b_hat_RN_future_estimate = est_remote_balances["R"]
        #
        # if self.verbose:
        #     print("Time {:.2f}: Estimate 5:".format(self.env.now))
        #     print("Time {:.2f}: b_hat_LN_future_estimate = {:.2f}, b_hat_RN_future_estimate = {:.2f}".format(self.env.now, self.b_hat_LN_future_estimate, self.b_hat_RN_future_estimate))



    # def max_swap_in_amount_allowed_by_on_chain_balance(self, neighbor):
    def max_swap_in_amount_allowed_by_on_chain_balance(self):
        # return min(self.on_chain_budget * (1 - self.rebalancing_parameters["server_swap_fee"]) - self.rebalancing_parameters["miner_fee"], self.capacities[neighbor])
        # return min(self.on_chain_budget * (1 - self.rebalancing_parameters["server_swap_fee"]) - self.rebalancing_parameters["miner_fee"], self.remote_balances[neighbor])
        return (self.on_chain_budget - self.rebalancing_parameters["miner_fee"]) / (1 + self.rebalancing_parameters["server_swap_fee"])

    def swap_in(self, neighbor, swap_amount, rebalance_request):
        # swap_amount is the net amount that is moving IN THE CHANNEL: it DOES NOT include the fees
        swap_in_fees = self.calculate_swap_in_fees(swap_amount)

        self.rebalancing_history_start_times.append(self.env.now)
        self.rebalancing_history_types.append(neighbor + "-IN")
        self.rebalancing_history_amounts.append(swap_amount)
        if self.verbose:
            print("Time {:.2f}: SWAP-IN requested in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))

        if swap_amount <= 0:
            if self.verbose:
                print("Time {:.2f}: SWAP-IN aborted due to violation of safety margin in channel N-{}.".format(self.env.now, neighbor))
            self.rebalancing_history_results.append("ABORTED")
            self.rebalancing_history_end_times.append(self.env.now)
        elif self.on_chain_budget < swap_amount + swap_in_fees:
            if self.verbose:
                print("Time {:.2f}: SWAP-IN aborted due to insufficient on-chain balance.".format(self.env.now, neighbor, swap_amount))
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
                self.swap_in_successful_in_channel.append(neighbor)

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
            print("Time {:.2f}: SWAP-OUT requested in channel N-{} with amount {:.2f}.".format(self.env.now, neighbor, swap_amount))

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
            self.swap_out_successful_in_channel.append(neighbor)

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

            # for neighbor in ["L", "R"]:
            #     self.env.process(self.perform_rebalancing_if_needed(neighbor))
            self.check_time_index += 1
            self.env.process(self.perform_rebalancing_if_needed())
                # yield self.env.process(self.perform_rebalancing_if_needed(neighbor))
            # yield self.env.process(self.perform_rebalancing_if_needed("L"))

            # if self.on_chain_budget > self.target_max_on_chain_amount:
            #     if self.verbose:
            #         print("Time {:.2f}: Maximum on-chain amount reached. Simulation is stopped.".format(self.env.now))
            #         print("Time {:.2f}: Final balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(
            #                 self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"],
            #                 self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"],
            #                 self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))
            #     break
