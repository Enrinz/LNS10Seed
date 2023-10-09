# -*- coding: utf-8 -*-
import abc
import random
import numpy as np
from abc import abstractmethod
from decimal import *
from solver.solution import Solution


class Destroy(object):
    """Base class for destroy operators"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, setting):
        self.setting = setting[0]
        self.evaluate_second_stage = setting[1]
        self.check_next_customer_infeasible = setting[2]
        self.check_route_feasible_after_insertion = setting[3]

    @abstractmethod
    def apply(self, solution : Solution):
        pass

    def __str__(self):
        return self.name

def _generalized_random_destroy(solution: Solution, setting):
    customers_to_remove = []
    prob_list = np.zeros(len(solution.routes))
    # for each route, compute the sampling probability
    for i, route in enumerate(solution.routes):
        customer_count = 0
        for element in route:
            if element["isCustomer"]:
                customer_count += 1
        res = customer_count / len(solution.instance.customers)
        prob_list[i] = res
    prob_list /= sum(prob_list)

    # remove random customers until length of customers_to_remove reaches gamma_c
    chosen_routes = np.random.choice(
        np.arange(len(prob_list)),
        p = prob_list,
        size = setting["gamma_c"], 
        replace = True, 
    )
    for route_idx in chosen_routes:
        # until a valid choice is made we repeat the sampling
        chosen_element = np.random.choice(np.arange(len(solution.routes[route_idx])))
        chosen_node = solution.routes[route_idx][chosen_element]
        while chosen_node["isDepot"]:
            chosen_element = np.random.choice(np.arange(len(solution.routes[route_idx])))
            chosen_node = solution.routes[route_idx][chosen_element]
        if chosen_node["isCustomer"] and (chosen_node["name"] not in customers_to_remove):
            customers_to_remove.append(chosen_node["name"])
    # remove selected nodes
    solution.remove_nodes_from_route(customers_to_remove, chosen_routes)
    return customers_to_remove
                    


class RandomDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "RandomDestroyCustomer"

    def apply(self, solution : Solution):                    
        return _generalized_random_destroy(solution, self.setting)


class WorstDistanceDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "WorstDistanceDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        # compute the cost to and from each node and save in distance_costs
        distance_costs = {}
        for route_idx, route in enumerate(solution.routes):
            for j, node in enumerate(route):
                if node["isCustomer"]:
                    cost_node_to = solution.instance.compute_arc_cost(route[j-1]["name"], route[j]["name"])
                    cost_node_from =  solution.instance.compute_arc_cost(route[j]["name"], route[j+1]["name"])
                    # save node name and route index
                    distance_costs[(node["name"], route_idx)] =  cost_node_to + cost_node_from
        # sort all the distances
        sorted_distance_costs = dict(sorted(distance_costs.items(), key = lambda item: item[1], reverse = True))
        routes_list_idx = []
        while len(customers_to_remove) < self.setting["gamma_c"]:
            # sort the cost in descending order, remove the most constly
            lamb = random.uniform(0, 1)
            position_to_remove = int(
                (lamb**self.setting["worst_removal_determinism_factor"]) * len(sorted_distance_costs)
            )
            customer_to_remove = list(sorted_distance_costs.keys())[position_to_remove]
            # if not present add node to remove
            if customer_to_remove[0] not in customers_to_remove:
                customers_to_remove.append(customer_to_remove[0])
                routes_list_idx.append(customer_to_remove[1])
        # remove nodes
        solution.remove_nodes_from_route(customers_to_remove, routes_list_idx)
        return customers_to_remove


class WorstTimeDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "WorstTimeDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []

        time_costs = {}
        for i, route in enumerate(solution.routes):
            for j, node in enumerate(route):
                if node["isCustomer"]:
                    # compute the time the arrival of the vehicle and the early customer window
                    cost = np.abs(solution.instance.customers_dict[node["name"]]["ReadyTime"] - solution.arrival_times[i][j])
                    time_costs[(node["name"], i)] = cost
        # sort them in decreasing order
        sorted_time_costs = dict(sorted(time_costs.items(), key = lambda item: item[1], reverse = True))
        routes_list_idx = []
        while len(customers_to_remove) < self.setting["gamma_c"]:
            # remove the most time consuming
            lamb = random.uniform(0, 1)
            position_to_remove = int(
                (lamb**self.setting["worst_removal_determinism_factor"]) * len(sorted_time_costs)
            )
            customer_to_remove = list(sorted_time_costs.keys())[position_to_remove]
            if customer_to_remove[0] not in customers_to_remove:
                customers_to_remove.append(customer_to_remove[0])
                routes_list_idx.append(customer_to_remove[1])
        # find customers in solution and remove them
        solution.remove_nodes_from_route(customers_to_remove, routes_list_idx)

        return customers_to_remove

def _apply_generalized_shaw(shaw_lambda, shaw_mu, shaw_nu, shaw_csai, solution : Solution, setting):
    route_idx = 0
    customers_to_remove = []
    shaw_lambda = shaw_lambda # distance relatedness
    shaw_mu = shaw_mu # time relatedness
    shaw_nu = shaw_nu # capacity relatedness
    shaw_csai = shaw_csai # possible serving vehicles relatedness
    # choose a random customer
    custom_selected = solution.instance.customers[
        random.randrange(0, len(solution.instance.customers), 1)
    ]
    customers_to_remove.append(custom_selected["StringID"])

    routes_names = solution.generate_route_id()
    relatedness = {}
    for customer in solution.instance.customers:
        # compute relatedness parameter between each customer and the random chosen one
        shaw_terms = 0
        if customer["StringID"] != custom_selected["StringID"]:
            shaw_terms += solution.instance.distance_matrix._get_item(
                custom_selected["StringID"],
                customer["StringID"]
            ) * shaw_lambda
            shaw_terms += np.abs(customer["ReadyTime"] - custom_selected["ReadyTime"]) * shaw_mu
            shaw_terms += np.abs(customer["demand"] - custom_selected["demand"]) * shaw_nu
            gammaij = 1
            for idx, route in enumerate(routes_names):
                if customer["StringID"] in route:
                    route_idx = idx
                if customer["StringID"] in route and custom_selected["StringID"] in route:
                    gammaij = -1
                
            shaw_terms += shaw_csai * gammaij
            relatedness.update({(customer["StringID"], route_idx) : shaw_terms})

    routes_list_idx = []
    while len(customers_to_remove) < setting["gamma_c"]:
        # remove the most related
        sorted_relatedness = dict(sorted(relatedness.items(), key = lambda item: item[1], reverse = True))
        lamb = random.uniform(0, 1)
        position_to_remove = int((lamb**setting["shaw_removal_determinism_factor"]) * len(sorted_relatedness))        
        customer_to_remove = list(sorted_relatedness.keys())[position_to_remove]
        if customer_to_remove[0] not in customers_to_remove:
            customers_to_remove.append(customer_to_remove[0])
            routes_list_idx.append(customer_to_remove[1])

    solution.remove_nodes_from_route(customers_to_remove, routes_list_idx)

    return customers_to_remove

class ShawDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ShawDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []
        shaw_lambda = self.setting["shaw_parameters"]["lambda"] # distance relatedness
        shaw_mu = self.setting["shaw_parameters"]["mu"] # time relatedness
        shaw_nu = self.setting["shaw_parameters"]["nu"] # capacity relatedness
        shaw_csai = self.setting["shaw_parameters"]["csai"] # possible serving vehicles relatedness

        customers_to_remove = _apply_generalized_shaw(
            shaw_lambda, shaw_mu, shaw_nu, shaw_csai, solution, self.setting
        )

        return customers_to_remove


class ProximityBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProximityBasedDestroyCustomer"

    def apply(self, solution : Solution):
        # identical to shaw removal, only the starting parameters change
        customers_to_remove = []
        shaw_lambda = self.setting["shaw_parameters"]["lambda"] # distance relatedness
        shaw_mu = 0 # time relatedness
        shaw_nu = 0 # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        customers_to_remove = _apply_generalized_shaw(
            shaw_lambda, shaw_mu, shaw_nu, shaw_csai, solution, self.setting
        )
        return customers_to_remove


class TimeBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "TimeBasedDestroyCustomer"
        

    def apply(self, solution : Solution):
        # identical to shaw removal, only the starting parameters change
        customers_to_remove = []
        shaw_lambda = 0 # distance relatedness
        shaw_mu = self.setting["shaw_parameters"]["mu"] # time relatedness
        shaw_nu = 0 # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        customers_to_remove = _apply_generalized_shaw(
            shaw_lambda, shaw_mu, shaw_nu, shaw_csai, solution, self.setting
        )

        return customers_to_remove


class DemandBasedDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "DemandBasedDestroyCustomer"

    def apply(self, solution : Solution):
        # identical to shaw removal, only the starting parameters change
        customers_to_remove = []
        shaw_lambda = 0 # distance relatedness
        shaw_mu = 0 # time relatedness
        shaw_nu = self.setting["shaw_parameters"]["nu"] # capacity relatedness
        shaw_csai = 0 # possible serving vehicles relatedness

        customers_to_remove = _apply_generalized_shaw(
            shaw_lambda, shaw_mu, shaw_nu, shaw_csai, solution, self.setting
        )

        return customers_to_remove



class ZoneDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ZoneDestroyCustomer"

    def apply(self, solution : Solution):
        customers_to_remove = []

        # choose a random zone
        which_zone = np.random.randint(0, len(solution.instance.rectangles))
        zone = solution.instance.rectangles[which_zone]
        routes_list_idx = []
        # try to remove as much as gamma_c customers from the solution
        for _ in range(self.setting["gamma_c"]):
            for customer in solution.instance.customers:
                # check for every customer if they are in the selected zone
                inzone = customer["x"] > zone[3][0] and customer["y"] > zone[3][1] and customer["x"] < zone[1][0] and customer["y"] < zone[1][1]
                if inzone and (customer["StringID"] not in customers_to_remove):
                    found = False
                    i = 0
                    # find idx of the route and save it, facilitates removal from the solution
                    while i < len(solution.routes) and not found:
                        for node in solution.routes[i]:
                             if customer["StringID"] == node["name"]:
                                routes_list_idx.append(i)
                                found = True
                                break
                        i += 1
                                
                    customers_to_remove.append(customer["StringID"])
                    break

        # find customers in solution and remove them
        solution.remove_nodes_from_route(customers_to_remove, routes_list_idx)

        return customers_to_remove


class RandomRouteDestroyCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "RandomRouteDestroyCustomer"

    def apply(self, solution : Solution):
        # choose a random route and remove customers from the route
        customers_to_remove = []
        route_idx = np.random.randint(0, len(solution.routes))
        route_to_remove = solution.routes[route_idx]
        # Try to select gamma_c customers from selected route 
        for _ in range(self.setting["gamma_c"]):
            customer_idx = np.random.randint(1, len(route_to_remove)-1)
            if route_to_remove[customer_idx]["isCustomer"] and route_to_remove[customer_idx]["name"] not in customers_to_remove:
                customers_to_remove.append(route_to_remove[customer_idx]["name"])
        
        # find customers in solution and remove them
        solution.remove_nodes_from_route(customers_to_remove, [route_idx])
        
        return customers_to_remove


class GreedyRouteRemoval(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "GreedyRouteRemoval"

    def apply(self, solution : Solution):
        # compute the cost of every route, remove customers from the most costly route
        customers_to_remove = []
        route_costs = np.zeros(len(solution.routes))
        for i, route in enumerate(solution.routes):
            route_cost = 0
            for j in range(len(route)-1):
                route_cost += solution.instance.compute_arc_cost(route[j]["name"], route[j+1]["name"])
            route_costs[i] = route_cost
        # consider most expensive route
        which_route = np.argmax(route_costs) # dict(sorted(route_costs.items(), key = lambda item: item[1], reverse = True))
        # add gamma_c customers from the route
        for i, idx in enumerate(solution.routes[which_route]):
            if idx["name"] not in customers_to_remove and idx["isCustomer"] and len(customers_to_remove) < self.setting["gamma_c"]:
                customers_to_remove.append(idx["name"])

        # find customers in solution and remove them
        solution.remove_nodes_from_route(customers_to_remove, [which_route])
        return customers_to_remove


class ProbabilisticWorstRemovalCustomer(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "ProbabilisticWorstRemovalCustomer"

    def apply(self, solution : Solution):
        # remove customers that would make the subsequent ones violated
        customers_to_remove = []
        stations_positions = []
        probability_infeasible_next_customer = {}

        for route_idx, route in enumerate(solution.routes):
            for node_idx, node in enumerate(route):
                if node["isStation"]:
                    stations_positions.append( (route_idx, node_idx) )
        # compute probability that next customers are violated
        for route_idx, station_idx in stations_positions:
            # generate delays experiment
            station_delays = sorted(np.random.exponential(
                1 / solution.instance.mu, 
                size = self.setting["n_scenarios"]
            ))

            for node_idx in range(station_idx + 1, len(solution.routes[route_idx])):
                if solution.routes[route_idx][node_idx]["isCustomer"]:
                    # compute probability that a customer makes the next infeasible
                    probability_infeasible_next_customer.update(
                        {
                            (solution.routes[route_idx][node_idx]["name"], route_idx) : self.check_next_customer_infeasible(
                                solution, 
                                [route_idx, node_idx], 
                                station_delays, 
                                station_idx
                            )
                        }
                    )

        # remove the most probable ones
        routes_list_idx = []
        sorted_prob = dict(sorted(probability_infeasible_next_customer.items(), key = lambda item: item[1], reverse = True))
        if len(sorted_prob) != 0:
            for customer in sorted_prob:
                if customer[0] not in customers_to_remove and len(customers_to_remove) < self.setting["gamma_c"]:
                    customers_to_remove.append(customer[0])
                    routes_list_idx.append(customer[1])
            # find customers in solution and remove them     
            solution.remove_nodes_from_route(customers_to_remove, routes_list_idx)
        else:
            # if no customers are found after station, apply the generalized random destroy operator instead
            customers_to_remove = _generalized_random_destroy(solution, self.setting)
        return customers_to_remove


class RandomDestroyStation(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "RandomDestroyStation"

    def apply(self, solution : Solution):
        stations_to_remove = []
        possible_stations = []
        station_name_to_route_idx = {}

        # retrieve all the stations
        for route_idx in range(len(solution.routes)):
            for node in solution.routes[route_idx]:
                if node["isStation"]:
                    possible_stations.append(node["name"])
                    station_name_to_route_idx.update({node["name"] : route_idx})
        # if there are more than gamma_s stations
        if len(possible_stations) > self.setting["gamma_s"]:
            # randomly select gamma_s stations
            stations_to_remove = np.random.choice(
                possible_stations,
                replace=False,
                size=self.setting["gamma_s"]
            )
        else:
            # else select all available stations
            stations_to_remove = possible_stations

        # save the route idx of the station, useful for faster removal of the nodes
        routes_list_idx = []
        for element in stations_to_remove:
            routes_list_idx.append(station_name_to_route_idx[element])

        # find stations in solution and remove them
        solution.remove_nodes_from_route(stations_to_remove, routes_list_idx)

        return stations_to_remove


class LongestWaitingTimeDestroyStation(Destroy):

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "LongestWaitingTimeDestroyStation"

    def apply(self, solution : Solution):
        # find the station which has the most waiting time and remove it
        stations_to_remove = []
        possible_stations = []
        routes_list_idx = [] 
        expected_waiting_times = {}
        # get station information
        for route_idx in range(len(solution.routes)):
            for node in solution.routes[route_idx]:
                if node["isStation"]:
                    possible_stations.append(node["name"])
                    expected_waiting_times.update(
                        {
                            (node["name"], route_idx) : solution.instance.get_expected_waiting_time(
                                solution.instance.stations_dict[node["name"]]["utilization_level"]
                            )
                        }
                    )
        # sort station by utilization_level
        sorted_station_times = dict(
            sorted(
                expected_waiting_times.items(), 
                key = lambda item: item[1], 
                reverse = True
            )
        )
        # add to stations_to_remove the stations with highest utilization_level
        if len(possible_stations) > self.setting["gamma_s"]:    
            for station in sorted_station_times:
                if len(stations_to_remove) < self.setting["gamma_s"]:
                    stations_to_remove.append(station[0])
                    routes_list_idx.append(station[1])
                else:
                    break
        else:
            for station in possible_stations:
                stations_to_remove.append(station[0])
                routes_list_idx.append(station[1])
        
        # find stations in solution and remove them
        solution.remove_nodes_from_route(stations_to_remove, routes_list_idx)

        return stations_to_remove