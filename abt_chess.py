from utils import nogood_isin_nogoods, normalize_nogood
from copy import deepcopy
import numpy as np

from chess import (
    king_rule_attack,
    knight_rule_attack,
    wbishop_rule_attack,
    bbishop_rule_attack,
)


class AgentABT(object):
    def __init__(
        self, number, value, universal_domain, attack_move, graph_size, figtype
    ):
        self.priority = number
        self.message_queue = []
        self.local_view = {}
        self.nogoods = []
        self.domain = universal_domain
        self.graph_size = graph_size
        self.attack_move = attack_move
        self.value = value
        self.figtype = figtype

    def add_neigbours(self, neighbors, lower_prio_neighbours):
        self.lower_prio_neighbours = lower_prio_neighbours
        self.neighbors = neighbors

    def get_constraint_one(self, value, other):

        other_value = other.value
        if self.attack_move(
            value[0], value[1], other_value[0], other_value[1], self.graph_size
        ):
            return True
        if other.attack_move(
            other_value[0], other_value[1], value[0], value[1], self.graph_size
        ):
            return True
        return False

    def get_constraint(self, value):
        """TRUE means BAD"""
        result = []
        for opponent in self.neighbors:
            result.append(self.get_constraint_one(value, opponent))
        result = bool(sum(result))
        return result

    def handle_message(self):
        if len(self.message_queue) > 0:
            message_type, message_value = self.message_queue.pop(0)

            if message_type == "ok":
                someone, val = message_value
                self.handle_ok(someone, val)
            elif message_type == "nogood":
                self.handle_nogood(message_value)
            else:
                self.handle_add_neighbour(message_value)

    def send_new_value(self):
        i_message = ["ok", [self, self.value]]
        for neighbour in self.lower_prio_neighbours:
            neighbour.message_queue.append(i_message)

    def handle_ok(self, agent, val):
        print("handle_ok", self.priority)
        self.local_view[agent] = val
        self.check_local_view()

    def check_local_view(self):
        print("check_localview", self.priority)
        if self.get_constraint(self.value):
            for domain in self.domain:
                good_position = not self.get_constraint(domain)
                if good_position:
                    good_domain = domain
                    break
            if not good_position:
                self.backtrack()
            else:
                self.value = good_domain
                self.send_new_value()

    def handle_nogood(self, nogood):
        print("handle_nogood", self.priority)
        # NOGOOD = List(List-of-pairs(agent, agents value np.ndarray))
        if not nogood_isin_nogoods(nogood, self.nogoods):
            self.nogoods.insert(0, nogood)
            for ng in nogood:
                nogood_agent, nogood_position = ng
                if nogood_agent not in self.neighbors:
                    self.neighbors.append(nogood_agent)
                    self.local_view[nogood_agent] = nogood_agent.value
                    nogood_agent.message_queue.append(["new_neighbour", self])

        # update lower-prio list
        for neighbor in self.neighbors:
            if (
                neighbor.priority > self.priority
            ) and neighbor not in self.lower_prio_neighbours:
                self.lower_prio_neighbours.append(neighbor)

        old_value = self.value
        self.check_local_view()
        if (old_value != self.value).any():
            self.send_new_value()

    def handle_add_neighbour(self, someone):
        if someone not in self.neighbors:
            self.neighbors.append(someone)

        for neighbor in self.neighbors:
            if (
                neighbor.priority > self.priority
            ) and neighbor not in self.lower_prio_neighbours:
                self.lower_prio_neighbours.append(neighbor)

        message = ["ok", [self, self.value]]
        someone.message_queue.append(message)

    def backtrack(self):
        # should be inconsistant subset of local_view
        print("backtrack", self.priority)
        nogood = [[k, v] for k, v in self.local_view.items()]
        if nogood:
            nogood = normalize_nogood(nogood)
        if not nogood_isin_nogoods(nogood, self.nogoods):
            if ([] in nogood) or nogood == []:
                return None
        self.nogoods.insert(0, nogood)

        agents = []
        if nogood:
            for ng in nogood:
                agents.append(ng[0])

            lowest_prio_agent = agents[0]
            for agent in agents:
                if agent.priority > lowest_prio_agent.priority:
                    lowest_prio_agent = agent

            lowest_prio_agent.message_queue.append(["nogood", nogood])
            self.local_view.pop(lowest_prio_agent)
            self.check_local_view()

    def __str__(self):
        return str(self.__dict__)


class ABTChess(object):
    def __init__(self, chess_class):
        self.graph = deepcopy(chess_class)
        self.n = len(chess_class.figures)
        universal_domain = [
            np.array([x, y])
            for x in range(0, self.graph.size)
            for y in range(0, self.graph.size)
        ]

        rules = {
            "king": king_rule_attack,
            "knight": knight_rule_attack,
            "bbishop": bbishop_rule_attack,
            "wbishop": wbishop_rule_attack,
        }

        # agents own
        self.agents = [
            AgentABT(
                i,
                self.graph.figures[i][0],
                universal_domain,
                rules[self.graph.figures[i][1]],
                self.graph.size,
                self.graph.figures[i][1],
            )
            for i in range(self.n)
        ]

        for k, agent in enumerate(self.agents):
            lower_prio_neighbours = self.agents[k + 1 :]
            neighbors = self.agents[:k] + self.agents[k + 1 :]
            agent.add_neigbours(neighbors, lower_prio_neighbours)

        #         for k, nogood in enumerate(self.nogoods):
        # init contraints
        #             pass

        for agent in self.agents:
            agent.send_new_value()

    def are_conflicts(self):
        result = []
        for i in range(len(self.agents)):
            result.append(self.agents[i].get_constraint(self.agents[i].value))

        return bool(sum(result))

    def run_colorization(self, n=1):
        while True:
            important_agents = [
                agent for agent in self.agents if len(agent.message_queue) > 0
            ]
            if len(important_agents) > 0:
                for agent in important_agents:
                    agent.handle_message()
            else:
                if self.are_conflicts():
                    print()
                break

            for i in range(len(self.agents)):
                self.graph.figures[i][0] = self.agents[i].value

        # check_last_time:
        if self.are_conflicts():
            print("========\n\n\n\n\n\n\nNo SOLUTION!\n\n\n\n\n\n\n========")
            return -1

        for i in range(len(self.agents)):
            print(self.agents[i].figtype, self.agents[i].value)
            self.graph.figures[i][0] = self.agents[i].value
        self.graph.show_field()

        return 0
