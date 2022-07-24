# class RL_connector:
# def __init__(self):
import sys


# def process_action_to_respect_constraints(raw_action, state, min_swap_out_amount):
def process_action_to_respect_constraints(raw_action, state, min_swap_out_amount, capacities, target_max_on_chain_amount):
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
        min_swap_out_amount=min_swap_out_amount,
        # estimates_of_future_remote_balances=
        capacities=capacities,
        target_max_on_chain_amount=target_max_on_chain_amount,
    ):
        processed_r_L = 0.0
    else:
        processed_r_L = r_L

    if not rebalancing_amount_respects_decoupled_constraints(
        r_amount=r_R,
        neighbor="R",
        state=state,
        min_swap_out_amount=min_swap_out_amount,
        # estimates_of_future_remote_balances=
        capacities=capacities,
        target_max_on_chain_amount=target_max_on_chain_amount,
    ):
        processed_r_R = 0.0
    else:
        processed_r_R = r_R

    return [processed_r_L, processed_r_R]


# def rebalancing_amount_respects_decoupled_constraints(r_amount, neighbor, state, min_swap_out_amount, estimates_of_future_remote_balances):
def rebalancing_amount_respects_decoupled_constraints(r_amount, neighbor, state, min_swap_out_amount, capacities, target_max_on_chain_amount):
    # local_balance = None
    local_balance_absolute = None
    if neighbor == "L":
        # local_balance = state[1]
        local_balance_absolute = state[1] * capacities[neighbor]
        # estimate_of_future_remote_balance = estimates_of_future_remote_balances[0]
    elif neighbor == "R":
        # local_balance = state[2]
        local_balance_absolute = state[2] * capacities[neighbor]
        # estimate_of_future_remote_balance = estimates_of_future_remote_balances[1]
    else:
        print("Error: invalid arguments in rebalancing_amount_respects_constraints.")
        exit(1)

    r_amount_absolute = r_amount * capacities[neighbor]

    # on_chain_balance = state[4]
    on_chain_balance_absolute = state[4] * target_max_on_chain_amount

    # if (- local_balance <= r_amount <= - min_swap_out_amount) or (0 <= r_amount <= min(on_chain_balance, estimate_of_future_remote_balance)):
    # if (- local_balance <= r_amount <= - min_swap_out_amount) or (0.0 <= r_amount <= min(on_chain_balance_absolute, capacities[neighbor])):
    if (- local_balance_absolute <= r_amount_absolute <= - min_swap_out_amount) or (0.0 <= r_amount_absolute <= min(on_chain_balance_absolute, capacities[neighbor])):
        constraints_are_respected = True
    else:
        constraints_are_respected = False

    return constraints_are_respected


def rebalancing_amounts_respect_joint_constraint(processed_action):

    [processed_r_L, processed_r_R] = processed_action
    if (processed_r_L > 0.0) and (processed_r_R > 0.0):
        constraint_is_respected = False
    else:
        constraint_is_respected = True

    return constraint_is_respected


def expand_action(action):
    r_L = action[0]
    r_R = action[1]
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
    