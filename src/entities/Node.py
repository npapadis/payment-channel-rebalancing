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
    def __init__(self, env, node_parameters, rebalancing_parameters, verbose, verbose_also_print_transactions):
        self.env = env
        self.local_balances = {"L": node_parameters["initial_balance_L"], "R": node_parameters["initial_balance_R"]}
        self.remote_balances = {"L": node_parameters["capacity_L"] - node_parameters["initial_balance_L"], "R": node_parameters["capacity_R"] - node_parameters["initial_balance_R"]}
        self.capacities = {"L": node_parameters["capacity_L"], "R": node_parameters["capacity_R"]}
        self.fees = [node_parameters["base_fee"], node_parameters["proportional_fee"]]
        self.on_chain_budget = node_parameters["on_chain_budget"]
        self.target_max_on_chain_amount = self.on_chain_budget * 30
        # self.target_max_on_chain_amount = 100 * max(self.capacities["L"], self.capacities["R"])
        self.initial_fortune = node_parameters["initial_balance_L"] + node_parameters["initial_balance_R"] + node_parameters["on_chain_budget"]

        self.swap_IN_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.swap_OUT_amounts_in_progress = {"L": 0.0, "R": 0.0}
        self.rebalancing_parameters = rebalancing_parameters
        self.verbose = verbose
        self.verbose_also_print_transactions = verbose_also_print_transactions
        self.latest_transactions_buffer_size = 100
        self.latest_transactions = []
        self.A_hat_net_LN = 0
        self.A_hat_net_NL = 0
        self.A_hat_net_NR = 0
        self.A_hat_net_RN = 0
        self.b_hat_LN_future_estimate = 0
        self.b_hat_RN_future_estimate = 0

        if self.rebalancing_parameters["rebalancing_policy"] == "SAC":
            self.learning_parameters = LearningParameters()
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
            self.writer = SummaryWriter('../runs/{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "autotune" if self.learning_parameters.automatic_entropy_tuning else "")) # Tensorboard setup
            self.min_rebalancing_percentage = 0.1
            self.swap_failure_penalty_coefficient = 20
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

        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: SUCCESS: Transaction {} processed.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def reject_transaction(self, t):
        t.status = "FAILED"
        if self.verbose and self.verbose_also_print_transactions:
            print("Time {:.2f}: FAILURE: Transaction {} rejected.".format(self.env.now, t))
            print("Time {:.2f}: New balances are: |L| {:.2f}---{:.2f} |N| {:.2f}---{:.2f} |R|, on-chain = {:.2f}, IN-pending = {:.2f}, OUT-pending = {:.2f}.".format(self.env.now, self.remote_balances["L"], self.local_balances["L"], self.local_balances["R"], self.remote_balances["R"], self.on_chain_budget, self.swap_IN_amounts_in_progress["L"] + self.swap_IN_amounts_in_progress["R"], self.swap_OUT_amounts_in_progress["L"] + self.swap_OUT_amounts_in_progress["R"]))

    def process_transaction(self, t):
        # Update the memory of the latest transactions
        if len(self.latest_transactions) >= self.latest_transactions_buffer_size:
            self.latest_transactions.pop(0)
        self.latest_transactions.append(t)

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
        self.fee_losses_since_last_check_time += fee_losses_of_this_transaction
        self.cumulative_fee_losses += fee_losses_of_this_transaction
        self.rebalancing_fees_over_time.append(self.rebalancing_fees_since_last_transaction)    # We register rebalancing fees only when the next transaction arrives, and not at the actual time the rebalancing starts.
        self.cumulative_rebalancing_fees += self.rebalancing_fees_since_last_transaction
        self.rebalancing_fees_since_last_transaction = 0.0


    # def perform_rebalancing_if_needed(self, neighbor):
    def perform_rebalancing_if_needed(self):
        if self.rebalancing_parameters["rebalancing_policy"] == "none":
            return

        # Updates estimates according to buffer
        duration_of_transactions_in_buffer = (self.latest_transactions[-1].time_of_arrival - self.latest_transactions[0].time_of_arrival) if len(self.latest_transactions) > 0 else np.Inf

        # Estimations according to the buffer of amounts added to the respective balance per minute (unit time)
        self.A_hat_net_NL = (
                            sum([t.amount for t in self.latest_transactions if t.source == "L"]) - sum([t.amount - self.calculate_relay_fees(t.amount) for t in self.latest_transactions if t.source == "R"])
                            ) / duration_of_transactions_in_buffer
        self.A_hat_net_LN = - self.A_hat_net_NL
        self.A_hat_net_NR = (
                            sum([t.amount for t in self.latest_transactions if t.source == "R"]) - sum([t.amount - self.calculate_relay_fees(t.amount) for t in self.latest_transactions if t.source == "L"])
                            ) / duration_of_transactions_in_buffer
        self.A_hat_net_RN = - self.A_hat_net_NR

        if self.rebalancing_parameters["rebalancing_policy"] in ["autoloop", "loopmax"]:
            for neighbor in ["L", "R"]:
                self.env.process(self.perform_rebalancing_if_needed_in_single_channel(neighbor))

        elif self.rebalancing_parameters["rebalancing_policy"] == "SAC":

            if self.verbose:
                print("Time {:.2f}: SWAP check performed for all channels.".format(self.env.now))

            # if (self.rebalancing_locks["L"].count == 0) and (self.rebalancing_locks["R"].count == 0):  # if no rebalancing is in progress in any channel
            with self.rebalancing_locks["L"].request() as rebalance_request_L, self.rebalancing_locks["R"].request() as rebalance_request_R:  # Generate a request event for all channel locks
                yield rebalance_request_L & rebalance_request_R

                # Update parameters of all the networks if needed
                if len(self.replay_memory) > self.learning_parameters.batch_size:
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

                self.b_hat_LN_future_estimate = max(0, min(self.remote_balances["L"] + self.A_hat_net_LN*self.rebalancing_parameters["T_conf"], self.capacities["L"]))
                self.b_hat_RN_future_estimate = max(0, min(self.remote_balances["R"] + self.A_hat_net_RN*self.rebalancing_parameters["T_conf"], self.capacities["R"]))

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

                processed_action = process_action_to_respect_constraints(   # processed_action = (r_L, r_R), inside constraint bounds or zero
                        raw_action=raw_action,
                        state=state,
                        N=self,
                    )
                # processed_action = raw_action       # TODO: check if needed
                # processed_action = process_action_to_be_more_than_min_rebalancing_percentage(
                #     raw_action=raw_action,
                #     N=self,
                # )

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
                # execute non-zero swaps
                if r_L > 0.0:
                    if r_R > 0.0:
                        # print("Unreachable point reached: swap-ins in both channels is not a valid swap combination.")
                        # exit(1)
                        yield self.env.process(self.swap_in(
                            neighbor="L",
                            swap_amount=r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        ) & self.env.process(self.swap_in(
                            neighbor="R",
                            swap_amount=r_R * self.capacities["R"],
                            rebalance_request=rebalance_request_R)
                        )
                    elif r_R == 0:
                        yield self.env.process(self.swap_in(
                            neighbor="L",
                            # swap_amount=r_L,
                            swap_amount=r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        )
                        if self.verbose:
                            print("Time {:.2f}: No SWAP performed in channel N-R.". format(self.env.now))
                    else:   # if r_R < 0.0
                        yield self.env.process(self.swap_in(
                            neighbor="L",
                            # swap_amount=r_L,
                            swap_amount=r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        ) & self.env.process(self.swap_out(
                            neighbor="R",
                            # swap_amount=-r_R,
                            swap_amount=-r_R * self.capacities["R"],
                            rebalance_request=rebalance_request_R)
                        )

                elif r_L == 0.0:
                    if self.verbose:
                        print("Time {:.2f}: No SWAP performed in channel N-L.".format(self.env.now))
                    if r_R > 0.0:
                        yield self.env.process(self.swap_in(
                            neighbor="R",
                            # swap_amount=r_R,
                            swap_amount=r_R * self.capacities["R"],
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
                            swap_amount=-r_R * self.capacities["R"],
                            rebalance_request=rebalance_request_R)
                        )
                else:   # if r_L < 0.0
                    if r_R > 0.0:
                        yield self.env.process(self.swap_out(
                            neighbor="L",
                            # swap_amount=-r_L,
                            swap_amount=-r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        ) & self.env.process(self.swap_in(
                            neighbor="R",
                            # swap_amount=r_R,
                            swap_amount=r_R * self.capacities["R"],
                            rebalance_request=rebalance_request_R)
                        )
                    elif r_R == 0:
                        yield self.env.process(self.swap_out(
                            neighbor="L",
                            # swap_amount=-r_L,
                            swap_amount=-r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        )
                        if self.verbose:
                            print("Time {:.2f}: No SWAP performed in channel N-R.".format(self.env.now))
                    else:  # if r_R < 0.0
                        yield self.env.process(self.swap_out(
                            neighbor="L",
                            # swap_amount=-r_L,
                            swap_amount=-r_L * self.capacities["L"],
                            rebalance_request=rebalance_request_L)
                        ) & self.env.process(self.swap_out(
                            neighbor="R",
                            # swap_amount=-r_R,
                            swap_amount=-r_R * self.capacities["R"],
                            rebalance_request=rebalance_request_R)
                        )

                self.b_hat_LN_future_estimate = max(0, min(self.remote_balances["L"] + self.A_hat_net_LN*self.rebalancing_parameters["T_conf"], self.capacities["L"]))
                self.b_hat_RN_future_estimate = max(0, min(self.remote_balances["R"] + self.A_hat_net_RN*self.rebalancing_parameters["T_conf"], self.capacities["R"]))
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

                # reward = - ( self.fee_losses_since_last_check_time
                #     +
                #     (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else 0) +
                #     self.calculate_swap_out_fees(r_L_out) +
                #     (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else 0) +
                #     self.calculate_swap_out_fees(r_R_out)
                # )
                [r_L_in, r_L_out, r_R_in, r_R_out] = expand_action(processed_action)   # expanded action = (r_L_in, r_L_out, r_R_in, r_R_out)
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
                # reward = - (self.fee_losses_since_last_check_time / (max(self.capacities["L"], self.capacities["R"])) +
                #             (self.calculate_swap_in_fees(r_L_in) / self.capacities["L"] if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in) / self.capacities["L"]) +
                #             (self.calculate_swap_out_fees(r_L_out) / self.capacities["L"] if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out) / self.capacities["L"]) +
                #             (self.calculate_swap_in_fees(r_R_in) / self.capacities["R"] if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in) / self.capacities["R"]) +
                #             (self.calculate_swap_out_fees(r_R_out) / self.capacities["R"] if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out) / self.capacities["R"])
                #             )
                # reward = - (self.fee_losses_since_last_check_time +
                #             (self.calculate_swap_in_fees(r_L_in) if ("L" in self.swap_in_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_L_in)) +
                #             (self.calculate_swap_out_fees(r_L_out) if ("L" in self.swap_out_successful_in_channel) else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_L_out)) +
                #             (self.calculate_swap_in_fees(r_R_in) if "R" in self.swap_in_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_in_fees(r_R_in)) +
                #             (self.calculate_swap_out_fees(r_R_out) if "R" in self.swap_out_successful_in_channel else self.swap_failure_penalty_coefficient * self.calculate_swap_out_fees(r_R_out))
                #             ) / (max(self.capacities["L"], self.capacities["R"]))

                total_fortune_before = next_state[1] + next_state[2] + next_state[4]
                total_fortune_after = state[1] + state[2] + state[4]
                reward = (total_fortune_after - total_fortune_before) / self.initial_fortune


                done = 0

                self.fee_losses_since_last_check_time = 0
                self.swap_in_successful_in_channel = []
                self.swap_out_successful_in_channel = []

                mask = float(not done)
                # self.replay_memory.push(state, processed_action, reward, next_state, mask)  # Append transition to memory
                self.replay_memory.push(state, raw_action, reward, next_state, mask)  # Append transition to memory

                if self.verbose:
                    print_precision = "%.5f"
                    state_printable = [print_precision % elem for elem in state]
                    processed_action_printable = [print_precision % elem for elem in processed_action]
                    raw_action_printable = [print_precision % elem for elem in raw_action]
                    next_state_printable = [print_precision % elem for elem in next_state]
                    print("Time {:.2f}: Tuple pushed to memory:\n\tstate = {}\n\tprocessed action = {} from raw action = {}\n\treward = {:.5f}\n\tnext state = {}".format(
                            self.env.now, state_printable, processed_action_printable, raw_action_printable, reward, next_state_printable
                        )
                    )
        else:
            print("{} is not a valid rebalancing policy.".format(self.rebalancing_parameters["rebalancing_policy"]))
            exit(1)


    def perform_rebalancing_if_needed_in_single_channel(self, neighbor):
        if self.verbose:
            print("Time {:.2f}: SWAP check performed for channel N-{}.".format(self.env.now, neighbor))

        if self.rebalancing_locks[neighbor].count == 0:  # if no rebalancing in progress in the N-neighbor channel
            with self.rebalancing_locks[neighbor].request() as rebalance_request:  # Generate a request event
                yield rebalance_request

                if self.rebalancing_parameters["rebalancing_policy"] == "autoloop":
                    midpoint = self.capacities[neighbor] * (self.rebalancing_parameters["lower_threshold"] + self.rebalancing_parameters["upper_threshold"]) / 2

                    if self.local_balances[neighbor] < self.rebalancing_parameters["lower_threshold"] * self.capacities[neighbor]:  # SWAP-IN
                        swap_amount = midpoint - self.local_balances[neighbor]
                        yield self.env.process(self.swap_in(neighbor, swap_amount, rebalance_request))
                    elif self.local_balances[neighbor] > self.rebalancing_parameters["upper_threshold"] * self.capacities[neighbor]:  # SWAP-OUT
                        swap_amount = self.local_balances[neighbor] - midpoint
                        yield self.env.process(self.swap_out(neighbor, swap_amount, rebalance_request))
                    else:
                        pass  # no rebalancing needed
                        if self.verbose:
                            print("Time {:.2f}: SWAP not needed in channel N-{}.".format(self.env.now, neighbor))
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

                    net_rate_of_local_balance_change = {"L": self.A_hat_net_NL, "R": self.A_hat_net_NR}

                    if net_rate_of_local_balance_change[neighbor] < 0:  # SWAP-IN
                        expected_time_to_depletion = self.local_balances[neighbor] / (- net_rate_of_local_balance_change[neighbor])
                        if expected_time_to_depletion - self.rebalancing_parameters["check_interval"] < self.rebalancing_parameters["T_conf"]:
                            safety_margin_in_coins = - net_rate_of_local_balance_change[neighbor] * self.rebalancing_parameters["safety_margins_in_minutes"][neighbor]
                            swap_amount = self.max_swap_in_amount(neighbor) - safety_margin_in_coins
                            # swap_amount = self.max_swap_in_amount(neighbor)
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
        self.rebalancing_history_amounts.append(swap_amount)
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
