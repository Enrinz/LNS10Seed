# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from simulator import *

class Instance:
    def __init__(self, configs: dict, simulator: Simulator):
        self.simulator = simulator
        self.cost_driver = configs["driver_wage"]
        self.cost_overtime = configs["overtime_cost_numerator"] / configs["overtime_cost_denominator"]
        self.cost_vehicle = configs["fixed_vehicle_acquisition"]
        self.cost_energy = configs["unit_energy_cost"]
        # station utilization level boundaries
        self.rho_low = configs["rho_low"]
        self.rho_high = configs["rho_high"]
        self.battery_tank_capacity = 0.0 # battery tank capacity
        self.vehicle_cargo_capacity = 0.0 # vehicle cargo capacity
        self.vehicle_consumption_rate = 0.0 # vehicle consumption rate
        self.vehicle_charging_rate = 0.0 # vehicle recharging rate
        self.average_velocity = 0.0

        self.customers = []
        self.charging_stations = []
        self.depot_point = {}
        
        self.base_electric_vehicle = {}

        self.read_from_solomon_instance(
            configs["main_path"],
            configs["instance_file_name"]
        )
        self.distance_matrix = DistanceMatrix(
            self.depot_point,
            self.customers,
            self.charging_stations
        )

        self.distance_matrix.generate_distance_matrix()
        self.initial_setup_electric_vehicle()
        self.setup_utilization_level()
        self.simulator.generate_service_time(self.customers)
        self.setup_customers_dict()
        self.setup_stations_dict()
        self.define_rectangles()
        self.E_recharge = 0.4 * self.vehicle_charging_rate * self.battery_tank_capacity
        self.mu = 1/self.E_recharge

    def define_rectangles(self):
        # find corner points
        xmin = 999
        ymin = 999
        xmax = 0
        ymax = 0
        for customer in self.customers:
            if customer["x"] < xmin:
                xmin = customer["x"]
            if customer["y"] < ymin:
                ymin = customer["y"]
            if customer["x"] > xmax:
                xmax = customer["x"]
            if customer["y"] > ymax:
                ymax = customer["y"]
        # compute mid points
        edgex = np.floor(np.abs(xmin-xmax)/2)
        edgey = np.floor(np.abs(ymin-ymax)/2)
        # generate 4 rectangles
        self.rectangles = [
            [(edgex, ymin), (xmin , edgey), (edgex, edgey), (xmin, ymin)],
            [(xmax, ymin), (xmax, edgey), (edgex, edgey), (edgex, ymin)],
            [(xmax, edgey), (xmax, ymax), (edgex, ymax), (edgex, edgey)],
            [(edgex, edgey), (edgex, ymax), (xmin, ymax), (xmin, edgey)]
        ]

    def read_from_solomon_instance(self, main_path, instance_file_name):
        file_in = open(
            os.path.join(
                main_path, "dataTestSeed",
                instance_file_name),
            "r"
        )
        lines = file_in.readlines()
        file_in.close()
        
        base_dict = {
            "StringID" : "",
            "Type" : "",
            "x" : "",
            "y" : "",
            "demand" : "",
            "ReadyTime" : "",
            "DueDate" : "",
            "ServiceTime" : ""
        }
        # remove first line:
        lines.pop(0)

        line = lines.pop(0).strip("\n")
        while line != "":
            line = line.split()
            tmp_dict = dict(base_dict)
            # complete base_dict for selected node
            for i, key in enumerate(tmp_dict.keys()):
                if key != "StringID" and key != "Type":
                    tmp_dict[key] = float(line[i])
                else:
                    tmp_dict[key] = line[i]
            # add tmp_dict in the right list
            if tmp_dict["Type"] == "c":
                self.customers.append(tmp_dict)
            elif tmp_dict["Type"] == "f":
                tmp_dict["utilization_level"] = 0.0
                self.charging_stations.append(tmp_dict)
            else:
                self.depot_point = dict(tmp_dict)

            line = lines.pop(0).strip("\n")
        
        self.battery_tank_capacity = float(lines.pop(0).strip("\n").split("/")[-2])
        self.vehicle_cargo_capacity = float(lines.pop(0).strip("\n").split("/")[-2])
        self.vehicle_consumption_rate = float(lines.pop(0).strip("\n").split("/")[-2])
        self.vehicle_charging_rate = float(lines.pop(0).strip("\n").split("/")[-2])
        self.average_velocity = float(lines.pop(0).strip("\n").split("/")[-2])
        
    def setup_customers_dict(self):
        self.customers_dict = {}
        for customer in self.customers:
            self.customers_dict.update({
                customer["StringID"] : {}
            })
            for field in customer:
                if field != "StringID":
                    self.customers_dict[customer["StringID"]].update({
                        field : customer[field]
                    })

    def setup_stations_dict(self):
        self.stations_dict = {}
        for station in self.charging_stations:
            self.stations_dict.update({
                station["StringID"] : {}
            })
            for field in station:
                if field != "StringID":
                    self.stations_dict[station["StringID"]].update({
                        field : station[field]
                    })



    def initial_setup_electric_vehicle(self):
        self.base_electric_vehicle["SoC"] = 1
        self.base_electric_vehicle["current_cargo"] = 0
        self.base_electric_vehicle["time"] = 0
        self.base_electric_vehicle["depot_closure"] = self.depot_point["DueDate"]
        self.base_electric_vehicle["SoC_list"] = [1]


    def get_expected_waiting_time(self, utilization_level):
        rho = utilization_level
        lambd = self.mu * rho
        return rho / (self.mu - lambd)

    def setup_utilization_level(self):
        for i in range(len(self.charging_stations)):
            self.charging_stations[i]["utilization_level"] = self.simulator.generate_utilization_level()

    def compute_arc_cost(self, node1, node2):
        cost = 0
        distance = self.distance_matrix._get_item(
            node1,
            node2
        )
        travel_time = distance/self.average_velocity
        cost += (distance * self.cost_energy)
        cost += (travel_time * self.cost_driver)
        return cost

    def compute_energy_cost(self, route, node_nmb = 0):
        route_distance = 0
        for i in range(node_nmb, len(route)-1):
            route_distance += self.distance_matrix._get_item(
                route[i]["name"],
                route[i+1]["name"]
            )
        route_energy = self.cost_energy * route_distance

        return route_energy


    def show_solution(self, filepath = None):
        g = nx.DiGraph()
        color_map = []

        for customer in self.instance.customers:
            g.add_node(customer["StringID"], pos = (customer["x"], customer["y"]))
            color_map.append('yellow')
        
        for station in self.instance.charging_stations:
            g.add_node(station["StringID"], pos = (station["x"], station["y"]))
            color_map.append('green')

        g.add_node(self.instance.start_point["StringID"], pos = (station["x"], station["y"]))
        color_map.append('red')

        edge_label = {}
        for j, route in enumerate(self.solution.routes):
            for i in range(len(route)-1):
                g.add_edge(route[i]["name"], route[i+1]["name"])
                edge_label.update({(route[i]["name"], route[i+1]["name"]) : str(j)})

        node_label = {}
        for j, route in enumerate(self.solution.routes):
            for i, element in enumerate(route):
                if i != 0:
                    node_label.update({element["name"] : str(int(self.solution.arrival_times[j][i-1]))})
        # TODO: cambiare colore per le varie routes

        pos = nx.get_node_attributes(g, 'pos')
        pos_nodes = self.nudge(pos, 0, -2.5)

        nx.draw(g, with_labels = True, pos = pos, node_color = color_map)
        nx.draw_networkx_edge_labels(g, pos = pos, edge_labels = edge_label)
        nx.draw_networkx_labels(g, pos = pos_nodes, labels = node_label)
        if filepath: # TODO: completare qui results
            pass# stampo su file
        else:
            plt.show()


    def nudge(self, pos, x_shift, y_shift):
        return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

    def plot_sample_stability(self, OF_list, current_dir, instance_name, recourse_combination, which_stability, show = False):
        x = range(1, OF_list + 1)
        y = []
        yerr = []
        for idx in range(OF_list):
            y.append(np.mean(OF_list[idx]))
            yerr.append(np.std(OF_list[idx]))
        
        plt.errorbar(x, y, yerr = yerr)
        plt.xlabel("Scenarios")
        plt.ylabel("OF")
        plt.title("In-Sample Stability")
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(
                current_dir,
                "results",
                "plot_"+which_stability+"_sample_"+str(instance_name)+"_"+recourse_combination[0]+"_"+recourse_combination[1]
                )
            )
            



class DistanceMatrix():

    def __init__(self, depot_point, customers, charging_stations):
        self.depot_point = depot_point
        self.customers = customers
        self.charging_stations = charging_stations

    def generate_distance_matrix(self):
        row_dict = {"StringID" : "", "x" : "", "y" : ""}
        row_dict["StringID"] = [self.depot_point["StringID"]]
        row_dict["x"] = [self.depot_point["x"]]
        row_dict["y"] = [self.depot_point["y"]]

        for station in self.charging_stations:
            row_dict["StringID"].append(station["StringID"])
            row_dict["x"].append(station["x"])
            row_dict["y"].append(station["y"])

        for customer in self.customers:
            row_dict["StringID"].append(customer["StringID"])
            row_dict["x"].append(customer["x"])
            row_dict["y"].append(customer["y"])

        tmp_df = pd.DataFrame(data = row_dict, columns = row_dict.keys())

        self.distance_matrix_df = pd.DataFrame(
            squareform(pdist(tmp_df.iloc[:, 1:])),
            columns=tmp_df.StringID.unique(),
            index=tmp_df.StringID.unique()
        )
        cols = self.distance_matrix_df.columns
        self.correspondence = dict(zip(cols, range(len(cols))))

        self.distance_matrix = self.distance_matrix_df.to_numpy()
        


    def _get_item(self, node_from, node_to):
        return self.distance_matrix[self.correspondence[node_from]][self.correspondence[node_to]]