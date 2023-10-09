# -*- coding: utf-8 -*-
import abc
import copy
from abc import abstractmethod
import numpy as np

from solver.solution import Solution


class Repair(object):
    """Base class for repair operators"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, setting):
        self.setting = setting[0]
        self.evaluate_second_stage = setting[1]
        self.check_next_customer_infeasible = setting[2]
        self.check_route_feasible_after_insertion = setting[3]

    @abstractmethod
    def apply(self, solution : Solution, removed_elements):
        pass

    def __str__(self):
        return self.name

class NaiveGreedyRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "NaiveGreedyRepairCustomer"

    def apply(self, solution : Solution, removed_elements):
        # generate list of costs of each route for faster cost check
        base_cost_list = []
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )

        for customer_to_insert in removed_elements:
            # intialization
            best_position = None
            best_route = None
            max_cost_difference = -np.Inf
            # find the best best position for the customer
            for route_idx, route in enumerate(solution.routes):
                for node_idx in range(len(route)):
                    if node_idx != 0:
                        # check for the difference in cost (the higher the value, the better)
                        solution.add_node(customer_to_insert, route_idx, node_idx)

                        cost_difference = base_cost_list[route_idx] - solution.compute_OF_route(route_idx)
                        # update best position and route if possible 
                        if cost_difference > max_cost_difference:
                            max_cost_difference = cost_difference
                            best_route = route_idx
                            best_position = node_idx
                            insertion_idx = customer_to_insert
                        # come back to the previous solution
                        solution.remove_node(route[node_idx], route_idx)
            # add best customer in the best position
            solution.add_node(insertion_idx, best_route, best_position)

class GreedyRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "GreedyRepairCustomer"

    def apply(self, solution : Solution, removed_elements):
        # generate list of costs of each route for faster cost check
        base_cost_list = []
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )
        customers_repaired = [False] * len(removed_elements)
        for _ in range(len(removed_elements)):
            # intialization
            best_position = None
            best_route = None
            max_cost_difference = -np.Inf
            # for each customer
            for c, customer_to_insert in enumerate(removed_elements):
                # if it has not been inserted in the solution
                if not customers_repaired[c]:
                    # find the best best position for the customer
                    for route_idx, route in enumerate(solution.routes):
                        for node_idx in range(len(route)):
                            if node_idx != 0:
                                # check for the difference in cost (the higher the value, the better)
                                solution.add_node(customer_to_insert, route_idx, node_idx)

                                cost_difference = base_cost_list[route_idx] - solution.compute_OF_route(route_idx)
                                # update best position and route if possible 
                                if cost_difference > max_cost_difference:
                                    max_cost_difference = cost_difference
                                    best_route = route_idx
                                    best_position = node_idx
                                    insertion_idx = customer_to_insert
                                    removed_idx = c
                                # come back to the previous solution
                                solution.remove_node(route[node_idx], route_idx)

            # add best customer in the best position
            solution.add_node(insertion_idx, best_route, best_position)
            # set customer to be added in solution
            customers_repaired[removed_idx] = True


class ProbabilisticGreedyRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticGreedyRepairCustomer"

    # @profile
    def apply(self, solution: Solution, removed_elements):
        base_cost_list = []
        # generate list of costs of each route for faster cost check
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )

        for customer_to_insert in removed_elements:
            # intialization
            best_position = None
            best_route = None
            max_cost_difference = -np.Inf
            # for each customer

            # find the best customer in the best spot
            for route_idx, route in enumerate(solution.routes):
                for node_idx in range(1, len(route)):
                    
                    solution.add_node(customer_to_insert, route_idx, node_idx)
                    recourse_cost = self.evaluate_second_stage(solution, route_idx)
                    # check for the difference in cost (the higher the value, the better)
                    cost_difference = base_cost_list[route_idx] - (
                        solution.compute_OF_route(route_idx) + recourse_cost
                    )
                    # update best position and route if possible 
                    if cost_difference > max_cost_difference:
                        max_cost_difference = cost_difference
                        best_route = route_idx
                        best_position = node_idx
                        insertion_idx = customer_to_insert
                    # come back to the previous solution
                    solution.remove_node(route[node_idx], route_idx)

            # add best customer in the best position
            solution.add_node(insertion_idx, best_route, best_position)



class ProbabilisticGreedyConfidenceRepairCustomer(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticGreedyConfidenceRepairCustomer"

    def apply(self, solution : Solution, removed_elements: list):
        base_cost_list = []
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )
        num_iterations = len(removed_elements) # number of stations that had been removed
        removed = [False] * len(removed_elements)
        
        for _ in range(num_iterations):
            insertion_costs = []
            insertion_failsafe = []
            best_spot = {}
            max_cost_difference = -1000000
            for c, customer_to_insert in enumerate(removed_elements):
                insertion_costs.append([])
                insertion_failsafe.append([])
                if not removed[c]:
                    for route_idx, route in enumerate(solution.routes):
                        for node_idx in range(1, len(route)):     
                            solution.add_node(customer_to_insert, route_idx, node_idx)
                            tmp = base_cost_list[route_idx] - solution.compute_OF_route(
                                route_idx
                                )

                            insertion_failsafe[c].append(
                                {
                                    "spot" : {"route" : route_idx,"position" : node_idx}, 
                                    "cost_difference" : tmp
                                }
                            )
                            if tmp > max_cost_difference:
                                max_cost_difference = tmp
                                best_spot["route"] = route_idx
                                best_spot["position"] = node_idx
                                insertion_idx = customer_to_insert

                            feasible_prob = self.check_route_feasible_after_insertion(solution, node_idx) 
                            if feasible_prob > self.setting["greedy_confidence"]:
                                insertion_costs[c].append(
                                    {
                                        "spot" : {"route" : route_idx,"position" : node_idx}, 
                                        "cost_difference" : tmp
                                    }
                                )
                                if tmp > max_cost_difference:
                                    max_cost_difference = tmp
                                    best_spot["route"] = route_idx
                                    best_spot["position"] = node_idx
                                    insertion_idx = customer_to_insert
                                

                            solution.remove_node(route[node_idx], route_idx)

                        if insertion_costs[c] == []:
                            insertion_failsafe[c].append(False)

            count = 0
            for c, spot in enumerate(insertion_failsafe):
                if spot != []:
                    if spot[-1] == False:
                        count += 1
                if removed[c]:
                    count += 1

            if count == len(insertion_costs):
                solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
                removed[removed_elements.index(insertion_idx)] = True
                    
            else:
                solution.add_node(insertion_idx, best_spot["route"], best_spot["position"])
                removed[removed_elements.index(insertion_idx)] = True


class DeterministicBestRepairStation(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "DeterministicBestRepairStation"

    def apply(self, solution : Solution, removed_elements):
        base_cost_list = []
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )
        removed = [False] * len(removed_elements)
        r = 0 # index of removed elements list

        routes = solution.routes


        for route_idx, route in enumerate(routes):
            entered = False
            for node_idx in range(len(route)):
                if solution.vehicles[route_idx]["SoC_list"][node_idx] < 0.1 and not entered and route[node_idx]["isCustomer"]:
                    customer_pos = [route_idx, node_idx]
                    entered = True

            feasible_position = False
            if entered:
                while not feasible_position and not all(removed):
                    insertion_costs = []
                    for station in solution.instance.charging_stations:
                        solution.add_node(
                            station["StringID"],
                            customer_pos[0], 
                            customer_pos[1]
                            )
                        insertion_costs.append(
                                        {
                                            "cost_difference" : base_cost_list[route_idx] - solution.compute_OF_route(route_idx),
                                            "name" : station["StringID"]
                                        }
                                    )

                        solution.remove_node(solution.routes[customer_pos[0]][customer_pos[1]], customer_pos[0])

                    if insertion_costs != []:

                        max_cost_difference = -1000000
                        for spot in insertion_costs:
                            if spot["cost_difference"] > max_cost_difference:
                                max_cost_difference = spot["cost_difference"]
                                insertion_idx = spot["name"]

                        solution.add_node(insertion_idx, customer_pos[0], customer_pos[1])

                        feasible_position = True
                        removed[r] = True
                        r += 1
                    else:
                        customer_pos[1] -= 1


class ProbabilisticBestRepairStation(Repair):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticBestRepairStation"

    def apply(self, solution : Solution, removed_elements):
        base_cost_list = []
        for route_idx in range(len(solution.routes)):
            base_cost_list.append(
                solution.compute_OF_route(
                    route_idx
                )
            )
        num_iterations = len(removed_elements)
        removed = [False] * len(removed_elements)
        r = 0

        routes = solution.routes

        for route_idx, route in enumerate(routes):
            entered = False
            for node_idx in range(len(route)):
                if solution.vehicles[route_idx]["SoC_list"][node_idx] < 0.1 and not entered and route[node_idx]["isCustomer"]:
                    customer_pos = [route_idx, node_idx]
                    entered = True

            feasible_position = False
            if entered:
                while not feasible_position  and not all(removed):
                    insertion_costs = []
                    max_cost_difference = -1000000
                    for station in solution.instance.charging_stations:
                        solution.add_node(
                            station["StringID"],
                            customer_pos[0], 
                            customer_pos[1]
                            )
                        recourse_cost = self.evaluate_second_stage(solution, route_idx)
                        insertion_costs.append(
                                        {
                                            "cost_difference" : base_cost_list[route_idx] - (
                                                solution.compute_OF_route(
                                                    route_idx
                                                    )+ recourse_cost
                                                ),
                                            "name" : station["StringID"]
                                        }
                                    )
                        cost_difference = base_cost_list[route_idx] - (
                            solution.compute_OF_route(
                                route_idx
                                )+ recourse_cost
                            )
                        if cost_difference > max_cost_difference:
                            max_cost_difference = cost_difference
                            insertion_idx = station["StringID"]

                        solution.remove_node(solution.routes[customer_pos[0]][customer_pos[1]], customer_pos[0])

                    if insertion_costs != []:

                        solution.add_node(insertion_idx, customer_pos[0], customer_pos[1])

                        feasible_position = True
                        removed[r] = True
                        r += 1
                    else:
                        customer_pos[1] -= 1