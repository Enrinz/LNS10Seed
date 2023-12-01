# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import numpy as np
import random
from instance import Instance
import matplotlib.pyplot as plt


class Solution:
    def __init__(self, instance: Instance):
        self.routes = [] # list of list of nodes of each route
        self.vehicles = [] # vehicle of each route with soc, time, cargo
        self.arrival_times = [] # arrival times at each node
        self.instance = instance # instance object
        self.charging_station_pos = [] # position of the charging station on the route
    
    def add_node(self, node : str, route_idx : int, position_idx : int):
        self.routes[route_idx].insert(position_idx, self.setup_node_attributes(node))
        self.update_route(route_idx)

    def remove_node(self, node : dict, route_idx : int):
        for node_idx in range(len(self.routes[route_idx])):
            if self.routes[route_idx][node_idx]["name"] == node["name"]:
                node_nmb = node_idx
                break
        self.routes[route_idx].remove(node)
        if node_nmb < len(self.routes[route_idx]):
            self.update_route(route_idx, node_nmb=node_nmb)
        else:
            self.update_route(route_idx)


    def remove_nodes_from_route(self, lst_nodes : list, lst_route_idx : list):
        for route_idx in lst_route_idx:
            self.routes[int(route_idx)] = [x for x in self.routes[int(route_idx)] if x['name'] not in lst_nodes]
            self.update_route(int(route_idx))

    def remove_route(self, route_idx):
        self.routes.pop(route_idx)
        self.vehicles.pop(route_idx)
        self.arrival_times.pop(route_idx)
        self.charging_station_pos.pop(route_idx)

    def add_new_route(self, starting_time=None):
        # start from depot
        self.routes.append(
            [{"name" : "D0", "isCustomer" : False, "isStation" : False, "isDepot" : True}]
        )
        # add new vehicle
        self.vehicles.append({
                "SoC": 1,
                "current_cargo": 0,
                "time": 0,
                "depot_closure": self.instance.depot_point["DueDate"],
                "SoC_list": [1],
            }
        )
        # set starting time of vehicles, if any
        if starting_time:
            self.vehicles[-1]["time"] = starting_time
        # add arrival times
        self.arrival_times.append([])
        # add station info
        self.charging_station_pos.append([])

    def add_built_route(self, route, vehicle, arrival_time, charging_station_pos):
        # add an already existing route to the solution
        self.routes.append(copy.deepcopy(route))
        self.generate_charging_station_pos_route(0)
        self.vehicles.append(vehicle)
        self.arrival_times.append(arrival_time)
        
        self.update_route(0)
        

    def compute_OF(self):
        of = 0
        tot_distance = 0 # distance travelled cost 
        tot_time = 0 # driver paid for the time cost 
        overtime = 0 # driver overtime pay 
        n_routes = 0 # vehicle acquisition cost 
        # computing distance, total time and overtime:
        for j, route in enumerate(self.routes):
            for i in range(len(route)-1):
                tot_distance += self.instance.distance_matrix._get_item(
                    route[i]["name"],
                    route[i+1]["name"]
                )
            overtime = self.arrival_times[j][-1] - self.instance.depot_point["DueDate"]
            tot_time += self.arrival_times[j][-1]
            if overtime >= 0:
                overtime += overtime
        # summing up all the terms
        n_routes = len(self.routes)
        of = (self.instance.cost_energy * tot_distance) + (self.instance.cost_driver * tot_time)
        of += (self.instance.cost_overtime * overtime) + (self.instance.cost_vehicle * n_routes)
        return of 

    def compute_OF_route(self, route_idx : int) -> float:
        of = 0
        tot_distance = 0 # distance travelled cost 
        tot_time = self.arrival_times[route_idx][-1] # driver paid for the time cost 
        # driver overtime pay 
        overtime = max(self.arrival_times[route_idx][-1] - self.instance.depot_point["DueDate"], 0)
        # computing distance, total time and overtime:
        for i in range(len(self.routes[route_idx])-1):
            tot_distance += self.instance.distance_matrix._get_item(
                self.routes[route_idx][i]["name"],
                self.routes[route_idx][i+1]["name"]
            )
        # summing up all the terms
        n_routes = 1
        of = (self.instance.cost_energy * tot_distance) + (self.instance.cost_driver * tot_time)
        of += (self.instance.cost_overtime * overtime) + (self.instance.cost_vehicle * n_routes)
        return of 

    def check_solution_feasibility(self):
        for vehicle in self.vehicles:
            if vehicle["SoC"] < 0.1:
                return False
            elif vehicle["current_cargo"] > self.instance.vehicle_cargo_capacity:
                return False
        return True

    def generate_route_id(self):
        # generate routes with only name on the nodes, useful for plot and recourse
        routes = []
        for i, route in enumerate(self.routes):
            routes.append([])
            for node in route:
                routes[i].append(node["name"])
        return routes

    def update_station_waiting_time(self, route_idx, old_waiting_time):
        # this method update the station waiting time and recalculate the times
        # if there is at least one recharging station
        self.generate_charging_station_pos_route(route_idx)
        for station_position in self.charging_station_pos[route_idx]:
            # get station info
            station = self.instance.stations_dict[self.routes[route_idx][station_position]["name"]]
            # check if the station is busy or free
            is_free = random.random() < station["utilization_level"]
            if is_free:
                new_waiting_time = 0
            else:
                new_waiting_time = np.random.exponential(1 / self.instance.mu)
                # update the arrival times of customer after recharging station
                for i in range(station_position, len(self.routes[route_idx])):
                    # (NB: station_index -1 because the depot has no arrival time)
                    self.arrival_times[route_idx][i - 1] += old_waiting_time - new_waiting_time
            
    def generate_charging_station_pos_route(self, route_idx: int):
        # generate the charging_station_pos which contains the position of the stations
        # used in the recourse
        self.charging_station_pos = [[]]* len(self.routes)
        for node_idx, node in enumerate(self.routes[route_idx]):
            if node["isStation"]:
                self.charging_station_pos[route_idx].append(node_idx)

    def remove_empty_routes(self):
        found = False
        for i, route in enumerate(self.routes):
            for node in route:
                if len(route) == 2 and node["isDepot"]:
                    empty_route = i
                    found = True
                elif len(route) == 3 and node["isStation"]:
                    empty_route = i
                    found = True
            if found:
                self.remove_route(empty_route)
                found = False
    
    def update_route(self, route_idx, node_nmb = 0, target_SoC = 0.9):
        
        self.vehicles[route_idx] = {
            "SoC": 1,
            "current_cargo": 0,
            "time": 0,
            "depot_closure": self.instance.depot_point["DueDate"],
            "SoC_list": self.vehicles[route_idx]["SoC_list"],
        }
        if node_nmb != 0:
            # remove all the element from the after the considered node
            for _ in range(node_nmb-1, len(self.arrival_times[route_idx])):
                self.arrival_times[route_idx].pop()


            # remove from soc list the elements after node_nmb
            for _ in range(node_nmb, len(self.vehicles[route_idx]["SoC_list"])):
                if node_nmb < len(self.arrival_times[route_idx]):
                    self.vehicles[route_idx]['SoC_list'].pop()


            # set the time of the vehicle at node_nmb + service time of node_nmb
            if 0 <= route_idx < len(self.arrival_times) and len(self.arrival_times[route_idx]) != 0 and self.routes[route_idx][node_nmb]["isCustomer"]:
                self.vehicles[route_idx]["time"] = self.arrival_times[route_idx][-1] + self.instance.customers_dict[self.routes[route_idx][node_nmb]["name"]]["ServiceTime"]
            else:
                pass #print("Invalid route index or empty arrival times list.")
            if len(self.arrival_times[route_idx]) != 0 and self.routes[route_idx][node_nmb]["isStation"]:
                self.vehicles[route_idx]["time"] = self.arrival_times[route_idx][-1] + self.instance.stations_dict[self.routes[route_idx][node_nmb]["name"]]["ServiceTime"]
            if len(self.arrival_times[route_idx]) != 0 and self.routes[route_idx][node_nmb]["isDepot"]:
                if self.routes[route_idx][node_nmb - 1]["isCustomer"]:
                    self.vehicles[route_idx]["time"] = self.arrival_times[route_idx][-1] + self.instance.customers_dict[self.routes[route_idx][node_nmb - 1]["name"]]["ServiceTime"]
                elif self.routes[route_idx][node_nmb - 1]["isStation"]:
                    self.vehicles[route_idx]["time"] = self.arrival_times[route_idx][-1] + self.instance.stations_dict[self.routes[route_idx][node_nmb - 1]["name"]]["ServiceTime"]
        else:
            self.arrival_times[route_idx] = []
            self.vehicles[route_idx]["SoC_list"] = [1]

        for i in range(node_nmb, len(self.routes[route_idx])):
            if i != 0:
            # for each node after the depot
                if self.routes[route_idx][i]["isCustomer"]:
                    self.update_ev_customer_pickup(route_idx, i)
                elif self.routes[route_idx][i]["isStation"]:
                    self.update_ev_station(route_idx, i, target_SoC)
                elif self.routes[route_idx][i]["isDepot"]:
                    self.update_ev_depot(route_idx, i)

    def clean_solution(self, route_idx):
        dupes_index = []
        for node in range(1, len(self.routes[route_idx]) - 1):
            tmp_name = self.routes[route_idx][node - 1]["name"]
            if tmp_name == self.routes[route_idx][node]["name"]:
                dupes_index.append(node)
        if len(dupes_index) != 0:
            dupes_names = []
            for dupe in dupes_index:
                dupes_names.append(self.routes[route_idx][dupe]["name"])
            self.remove_nodes_from_route(dupes_names, [route_idx])
            

    def setup_node_attributes(self, node):
        attributes = {}
        attributes["name"] = node
        if "C" in node:
            attributes.update({"isCustomer" : True, "isStation" : False, "isDepot" : False})
        elif "S" in node:
            attributes.update({"isCustomer" : False, "isStation" : True, "isDepot" : False})
        else:
            attributes.update({"isCustomer" : False, "isStation" : False, "isDepot" : True})
        return attributes

    def update_ev_customer_pickup(self, route_idx, position_idx):
        # get customer dict (with time windows, etc.)
        customer = self.instance.customers_dict[self.routes[route_idx][position_idx]["name"]]
        # get distance to node in position_idx
        distance_travelled = self.instance.distance_matrix._get_item(
            self.routes[route_idx][position_idx-1]["name"],
            self.routes[route_idx][position_idx]["name"]
        )
        # update SoC
        self.vehicles[route_idx]["SoC"] -= distance_travelled * self.instance.vehicle_consumption_rate / self.instance.battery_tank_capacity
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        self.vehicles[route_idx]["current_cargo"] += customer["demand"]
        # if early arrival
        current_time = distance_travelled / self.instance.average_velocity + self.vehicles[route_idx]["time"]
        if current_time < customer["ReadyTime"]:
            self.arrival_times[route_idx].append(customer["ReadyTime"])
            self.vehicles[route_idx]["time"] = customer["ReadyTime"] + customer["ServiceTime"]
        else:
            self.arrival_times[route_idx].append(current_time)
            self.vehicles[route_idx]["time"] = current_time + customer["ServiceTime"]

    def update_ev_station(self, route_idx, position_idx, target_SoC):
        # compute distance
        distance_travelled = self.instance.distance_matrix._get_item(
            self.routes[route_idx][position_idx-1]["name"],
            self.routes[route_idx][position_idx]["name"]
        )
        # update SoC
        self.vehicles[route_idx]["SoC"] -= distance_travelled * self.instance.vehicle_consumption_rate / self.instance.battery_tank_capacity
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        # get expected waiting time (first stage computation)
        expected_waiting_time = self.instance.get_expected_waiting_time(
            self.instance.stations_dict[self.routes[route_idx][position_idx]["name"]]["utilization_level"]
        )
        # update SoC (after recharge)
        past_SoC = self.vehicles[route_idx]["SoC"]
        self.vehicles[route_idx]["SoC"] = target_SoC
        # update time
        self.vehicles[route_idx]["time"] += distance_travelled / self.instance.average_velocity
        self.arrival_times[route_idx].append(
            self.vehicles[route_idx]["time"]
        )
        # update time with recharge
        self.vehicles[route_idx]["time"] += expected_waiting_time + (self.instance.E_recharge * ( target_SoC - past_SoC ) / 0.4)

    def update_ev_depot(self, route_idx, position_idx):
        # compute distance
        distance_travelled = self.instance.distance_matrix._get_item(
            self.routes[route_idx][position_idx-1]["name"], 
            self.routes[route_idx][position_idx]["name"]
        )
        # update SoC
        self.vehicles[route_idx]["SoC"] -= self.instance.vehicle_consumption_rate * distance_travelled / self.instance.battery_tank_capacity
        self.vehicles[route_idx]["SoC_list"].append(self.vehicles[route_idx]["SoC"])
        # update time
        self.vehicles[route_idx]["time"] += distance_travelled / self.instance.average_velocity
        if position_idx != 0:
            self.arrival_times[route_idx].append(self.vehicles[route_idx]["time"])

    def plot_soc_history(self):
        for i, ele in enumerate(self.vehicles):
            plt.plot(ele["SoC_list"], label=f"Vehicle {i}")
            plt.xlabel('pick up')
            plt.ylabel('SoC')