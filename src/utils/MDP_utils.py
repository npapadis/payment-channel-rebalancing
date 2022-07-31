# class RL_connector:
# def __init__(self):
import sys


# def process_action_to_respect_constraints(raw_action, state, min_swap_out_amount):
def process_action_to_respect_constraints(raw_action, state, N):
    """
    Input:
        raw_action = (r_L, r_R), possibly off-constraint bounds, normalized
        state: normalized
        min_swap_out_amount: absolute
        capacities: absolute
        target_max_on_chain_amount: absolute
    Output: processed_action = (r_L, r_R), inside constraint bounds or zero, normalized
    """

    r_L = raw_action[0]
    r_R = raw_action[1]
    if not rebalancing_amount_respects_decoupled_constraints(
        r_amount=r_L,
        neighbor="L",
        state=state,
        N=N,
    ):
        processed_r_L = 0.0
    else:
        processed_r_L = r_L

    if not rebalancing_amount_respects_decoupled_constraints(
        r_amount=r_R,
        neighbor="R",
        state=state,
        N=N,
    ):
        processed_r_R = 0.0
    else:
        processed_r_R = r_R

    on_chain_balance_absolute = state[4] * N.target_max_on_chain_amount
    if not rebalancing_amounts_respect_coupled_constraint(
            r_L=processed_r_L,
            r_R=processed_r_R,
            on_chain_balance_absolute=on_chain_balance_absolute,
            N=N):
        if (processed_r_L > 0.0) and (processed_r_R > 0.0):
            r_L_in_alone_respects_on_chain_constraint = rebalancing_amounts_respect_coupled_constraint(r_L=processed_r_L, r_R=0, on_chain_balance_absolute=on_chain_balance_absolute, N=N)
            r_R_in_alone_respects_on_chain_constraint = rebalancing_amounts_respect_coupled_constraint(r_L=0, r_R=processed_r_R, on_chain_balance_absolute=on_chain_balance_absolute, N=N)
            if processed_r_L > processed_r_R:
                if r_L_in_alone_respects_on_chain_constraint:
                    processed_r_R = 0.0
                elif r_R_in_alone_respects_on_chain_constraint:
                    processed_r_L = 0.0
                else:
                    processed_r_L = 0.0
                    processed_r_R = 0.0
            else:   # if processed_r_L <= processed_r_R
                if r_R_in_alone_respects_on_chain_constraint:
                    processed_r_L = 0.0
                elif r_L_in_alone_respects_on_chain_constraint:
                    processed_r_R = 0.0
                else:
                    processed_r_L = 0.0
                    processed_r_R = 0.0

        elif (processed_r_L > 0.0) and (processed_r_R == 0.0):
            processed_r_L = 0.0
        elif (processed_r_L == 0.0) and (processed_r_R > 0.0):
            processed_r_R = 0.0
        else:
            pass

    return [processed_r_L, processed_r_R]


# def rebalancing_amount_respects_decoupled_constraints(r_amount, neighbor, state, min_swap_out_amount, estimates_of_future_remote_balances):
# def rebalancing_amount_respects_decoupled_constraints(r_amount, neighbor, state, min_swap_out_amount, capacities, target_max_on_chain_amount):
def rebalancing_amount_respects_decoupled_constraints(r_amount, neighbor, state, N):
    # local_balance = None
    local_balance_absolute = None
    if neighbor == "L":
        # local_balance = state[1]
        local_balance_absolute = state[1] * N.capacities[neighbor]
        # estimate_of_future_remote_balance = estimates_of_future_remote_balances[0]
    elif neighbor == "R":
        # local_balance = state[2]
        local_balance_absolute = state[2] * N.capacities[neighbor]
        # estimate_of_future_remote_balance = estimates_of_future_remote_balances[1]
    else:
        print("Error: invalid arguments in rebalancing_amount_respects_constraints.")
        exit(1)

    r_amount_absolute = r_amount * N.capacities[neighbor]

    # on_chain_balance = state[4]
    on_chain_balance_bound_absolute = state[4] * N.target_max_on_chain_amount - N.calculate_swap_in_fees(r_amount)

    # if (- local_balance <= r_amount <= - min_swap_out_amount) or (0 <= r_amount <= min(on_chain_balance, estimate_of_future_remote_balance)):
    # if (- local_balance <= r_amount <= - min_swap_out_amount) or (0.0 <= r_amount <= min(on_chain_balance_absolute, capacities[neighbor])):
    # if (- local_balance_absolute <= r_amount_absolute <= - N.min_swap_out_amount) or (0.0 <= r_amount_absolute <= min(on_chain_balance_bound_absolute, N.capacities[neighbor])):
    if (- local_balance_absolute <= r_amount_absolute <= - N.min_swap_out_amount) or (0.0 <= r_amount_absolute <= N.capacities[neighbor]):
        constraints_are_respected = True
    else:
        constraints_are_respected = False

    return constraints_are_respected


def rebalancing_amounts_respect_coupled_constraint(r_L, r_R, on_chain_balance_absolute, N):

    if (r_L > 0) and (r_R > 0):
        constraint_is_respected = True if (r_L + N.calculate_swap_in_fees(r_L) + r_R + N.calculate_swap_in_fees(r_R) <= on_chain_balance_absolute) else False
    elif (r_L > 0) and (r_R == 0):
        constraint_is_respected = True if (r_L + N.calculate_swap_in_fees(r_L) <= on_chain_balance_absolute) else False
    elif (r_L == 0) and (r_R > 0):
        constraint_is_respected = True if (r_R + N.calculate_swap_in_fees(r_R) <= on_chain_balance_absolute) else False
    else:
        constraint_is_respected = True

    return constraint_is_respected


def rebalancing_amounts_not_both_positive(processed_action):

    [processed_r_L, processed_r_R] = processed_action
    if (processed_r_L > 0.0) and (processed_r_R > 0.0):
        constraint_is_respected = False
    else:
        constraint_is_respected = True

    return constraint_is_respected


def expand_action(action):
    [r_L, r_R] = action
    expanded_action = [0.0, 0.0, 0.0, 0.0]     # expanded actions = (r_L_in, r_L_out, r_R_in, r_R_out)

    if r_L > 0.0:  # r_L > 0 ==> r_L_in = r_L
        expanded_action[0] = r_L
    elif r_L < 0.0:    # r_L < 0 ==> r_L_out = - r_L
        expanded_action[1] = - r_L
    else:
        pass

    if r_R > 0.0:
        expanded_action[2] = r_R
    elif r_R < 0.0:
        expanded_action[3] = - r_R
    else:
        pass

    return expanded_action


def process_action_to_be_more_than_min_rebalancing_percentage_v1(raw_action, N):
    # Version when action represents percentage of total channel capacity
    [r_L, r_R] = raw_action
    processed_r_L = 0.0 if (- N.min_swap_threshold_as_percentage_of_capacity <= r_L <= N.min_swap_threshold_as_percentage_of_capacity) else r_L
    processed_r_R = 0.0 if (- N.min_swap_threshold_as_percentage_of_capacity <= r_R <= N.min_swap_threshold_as_percentage_of_capacity) else r_R
    return [processed_r_L, processed_r_R]


def process_action_to_be_more_than_min_rebalancing_percentage_v2(raw_action, N):
    # Version when action represents percentage of liquidity available due to current constraints
    [r_L, r_R] = raw_action

    processed_r_L = 0.0 if (
                               ((r_L < 0.0) and (- N.min_swap_threshold_as_percentage_of_capacity * N.capacities["L"] <= r_L * N.max_swap_out_amount_due_to_current_constraints["L"]))
                               or ((r_L >= 0.0) and (r_L * N.max_swap_in_amount_due_to_current_constraints["L"] <= N.min_swap_threshold_as_percentage_of_capacity * N.capacities["L"]))
    ) else r_L

    processed_r_R = 0.0 if (
            ((r_R < 0.0) and (- N.min_swap_threshold_as_percentage_of_capacity * N.capacities["R"] <= r_R * N.max_swap_out_amount_due_to_current_constraints["R"]))
            or ((r_R >= 0.0) and (r_R * N.max_swap_in_amount_due_to_current_constraints["R"] <= N.min_swap_threshold_as_percentage_of_capacity * N.capacities["R"]))
    ) else r_R

    return [processed_r_L, processed_r_R]


class LearningParameters:   # args in original code's main.py
    def __init__(self):
        self.policy = "Gaussian"                    # Policy Type: Gaussian | Deterministic
        # self.eval = False                         # Evaluates a policy a policy every 10 episodes
        self.gamma = 0.99                           # Discount factor for reward
        self.tau = 0.005                            # Target smoothing coefficient (τ)
        self.lr = 0.0003                            # Learning rate
        self.alpha = 0.02                            # Temperature parameter α determines the relative importance of the entropy term against the reward
        self.automatic_entropy_tuning = False       # Automatically adjust α
        self.seed = 123456                          # Random seed
        self.batch_size = 1                         # Batch size
        # self.num_steps =                          # Maximum number of steps   ---> not needed, as simulator terminates based on time
        self.hidden_size = 256                      # Hidden size
        self.updates_per_step = 1                   # Model updates per simulator step
        self.start_steps = 10                       # Number of steps in the beginning for which we sample random actions and not from the learned distribution
        self.target_update_interval = 1             # Value target update per number of updates per step
        self.replay_size = 100                      # Size of replay buffer
        self.cuda = False                           # Run on CUDA

        self.nn_update_interval = 1

        self.on_chain_normalization_multiplier = 30
        self.episode_duration = 99
        self.target_local_balance_fractions_at_state_reset = {"L": 0.5, "R": 0.5}   # times the capacity of each channel
        # self.target_local_balance_fractions_at_state_reset = {"L": 0.0, "R": 1.0}   # times the capacity of each channel
        self.min_swap_threshold_as_percentage_of_capacity = 0.3
        self.swap_failure_penalty_coefficient = 20
