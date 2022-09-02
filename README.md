# Deep Reinforcement Learning-based rebalancing for profit maximization of relay nodes in Payment Channel Networks

This package is a Python SimPy-based Discrete Event Simulator for payment scheduling in a payment channel.
It simulates a relay PCN node with two payment channels, forwarding traffic from left to right and from right to left, and allows for experiments with various rebalancing policies based on the rebalancing mechanism of submarine swaps.
Transactions are generated from both sides according to customizable distributions of amounts and arrival times. Rebalancing operations are dispatched according to the chosen rebalancing policy.

The user can choose:
* the initial channel balances and the on-chain budget
* the transaction generation parameters: total transactions from each side, amount distribution (constant, uniform, gaussian, pareto, empirical from csv file), interarrival time distribution (exponential with customizable parameter)
* the rebalancing policy
* the diffent fees involved (base and proportional relay fee, server swap fee, miner fee)
* the time every which the node will check for the need for rebalancing, and the time for a swap to complete
* the number of experiments over which to calculate the average metrics
* the output filename

There are several rebalancing policies the user can choose from:
* `Autoloop`: a heuristic policy based on low and high thresholds currently used in practice
* `Loopmax`: a heuristic policy based on the expected demand calculated from empirical data and trying to rebalance as infrequently as possible and with the maximum amount possible.
* `RebEL`: "Rebalancing Enabled by Learning": a Deep Reinforcement Learning-based policy using [this implementation of Soft Actor-Critic](https://github.com/pranz24/pytorch-soft-actor-critic) that learns to perform approximately optimal rebalancing actions based on the observed demand from both sides.