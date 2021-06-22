import sys


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
        self.cleared = self.env.event()

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
            # yield self.env.process(self.current_node.process_transaction(self))                         # Once the channel belongs to the transaction, try to process it.
            self.current_node.process_transaction(self)                         # Once the channel belongs to the transaction, try to process it.

    def get_transaction_signature(self):
        transaction_signature = (self.time_of_arrival, self.source, self.destination, self.amount, self.status)
        return transaction_signature

    def __repr__(self):
        return "%s->%s t=%.2f a=%d" % (self.source, self.destination, self.time_of_arrival, self.amount)