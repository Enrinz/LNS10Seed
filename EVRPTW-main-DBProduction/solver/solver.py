# -*- coding: utf-8 -*-
import copy
import csv
import statistics
import math
import time
import random
from xml.dom.minidom import CharacterData
import numpy as np
from sklearn.decomposition import dict_learning
from instance import *
import simulator
from simulator.simulator import Simulator
from solver.ALNS_operators import *
from solver.plotter.plotter import Plotter
from solver.solution import Solution
import os
import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from datetime import datetime

path_soluzione_iniziale="soluzione iniziale.txt"
path_append="soluzione append.txt"

def clean(content):
    X=content.replace('\'','').replace('[','').replace(']','').replace(',','')
    print(X)
    a=[]
    a.append(X)

    tokenizer = Tokenizer(num_words=150)
    tokenizer.fit_on_texts(a)

    maxlen = 270
    a = tokenizer.texts_to_sequences(a)
    a = pad_sequences(a, padding='post', maxlen=maxlen)
    #print(a[0])
    return a[0]
moves_list = {
        "station" : {
            "destroy" : {
                0 : "RandomDestroyStation",
                1 : "LongestWaitingTimeDestroyStation",
                999 : "null"
            },
            "repair" : {
                0 : "DeterministicBestRepairStation",
                1 : "ProbabilisticBestRepairStation",
                999 : "null"
            }
        },
        "customer" : {
            "destroy" : {
                0 : "RandomDestroyCustomer",
                1 : "WorstDistanceDestroyCustomer",
                2 : "WorstTimeDestroyCustomer",
                3 : "RandomRouteDestroyCustomer",
                4 : "ZoneDestroyCustomer",
                5 : "DemandBasedDestroyCustomer",
                6 : "TimeBasedDestroyCustomer",
                7 : "ProximityBasedDestroyCustomer",
                8 : "ShawDestroyCustomer",
                9 : "GreedyRouteRemoval",
                10 : "ProbabilisticWorstRemovalCustomer",
                999 : "null"
            },
            "repair" : {
                0 : "GreedyRepairCustomer",
                1 : "ProbabilisticGreedyRepairCustomer",
                2 : "NaiveGreedyRepairCustomer",
                999 : "null"
            }
        }
    }

class Solver:
    def __init__(self, configs: dict, instance: Instance):
        self.instance = instance
        self.configs = configs
        self.operators_setup()
        if self.configs["consider_pickup"] == "Yes":
            self.consider_pickup = True
        else:
            self.consider_pickup = False
        if self.configs["consider_partial_recharge"] == "Yes":
            self.consider_partial_recharge= True
        else:
            self.consider_partial_recharge= False
        

    def operators_setup(self):
        operators_configs = [
            self.configs,
            self.evaluate_second_stage,
            self.check_next_customer_infeasible,
            self.check_route_feasible_after_insertion,
        ]
        customer_repair_operators = [
            GreedyRepairCustomer(operators_configs),
            ProbabilisticGreedyRepairCustomer(operators_configs),
            NaiveGreedyRepairCustomer(operators_configs)
        ]
        customer_destroy_operators = [
            RandomDestroyCustomer(operators_configs),
            WorstDistanceDestroyCustomer(operators_configs),
            WorstTimeDestroyCustomer(operators_configs),
            RandomRouteDestroyCustomer(operators_configs),
            ZoneDestroyCustomer(operators_configs),
            DemandBasedDestroyCustomer(operators_configs),
            TimeBasedDestroyCustomer(operators_configs),
            ProximityBasedDestroyCustomer(operators_configs),
            ShawDestroyCustomer(operators_configs),
            GreedyRouteRemoval(operators_configs),
            ProbabilisticWorstRemovalCustomer(operators_configs)
        ]
        station_repair_operators = [
            DeterministicBestRepairStation(operators_configs),
            ProbabilisticBestRepairStation(operators_configs)
        ]
        station_destroy_operators = [
            RandomDestroyStation(operators_configs),
            LongestWaitingTimeDestroyStation(operators_configs)
        ]
        # define params alns
        self.operators = {
            "customer_repair" : {
                "score" : [0] * len(customer_repair_operators) ,
                "weight" : [1/len(customer_repair_operators)] * len(customer_repair_operators),
                "operators" : customer_repair_operators
            },
            "customer_destroy" : {
                "score" : [0] * len(customer_destroy_operators),
                "weight" : [1/len(customer_destroy_operators)] * len(customer_destroy_operators),
                "operators" : customer_destroy_operators
            },
            "station_repair" : {
                "score" : [0] * len(station_repair_operators),
                "weight" : [1/len(station_repair_operators)] * len(station_repair_operators),
                "operators" : station_repair_operators
            },
            "station_destroy" : {
                "score" : [0] * len(station_destroy_operators),
                "weight" : [1/len(station_destroy_operators)] * len(station_destroy_operators),
                "operators" : station_destroy_operators
            }
        }

    def solve(self, ran_seed, collect_data=False, verbose=False, initial_solution:Solution = None):
        max_retry = 100
        # set recourse variables based on configs
        it = 0
        T = self.configs["annealing_parameters"]["T_0"]
        # if initial_solution is specified, use that for x_0, otherwise generate it with the method
        if initial_solution is not None:
            x_0 = copy.deepcopy(initial_solution)
        else:
            feasible = False
            while not feasible and it < max_retry:
                x_0 = self.construct_initial_solution()
                if x_0.check_solution_feasibility():
                    feasible = True
                else:
                    # Monitor: it should not happen
                    raise ValueError("Initial Solution Not Feasible")
        # plot the initial solution
        if verbose:
            self.instance.show_solution()

        # setup solver solution variables

        x_best = copy.deepcopy(x_0)
        x_current = copy.deepcopy(x_0)
        x_previous = copy.deepcopy(x_0)

        OF_x_0 = x_0.compute_OF()
        OF_x_best = OF_x_0
        OF_x_current = OF_x_0
        OF_x_previous = OF_x_0

        write_final_row = {
            "Instance's Name" : "",
            "Iteration": "",
            "Seed" : "",
            "Initial Solution": "",
            "OFIS" : "",
            "Moves" : "",
            "OFFS" : "",
            "OF_Diff" : "",
            "Exe_Time_d-r" : "",
            "Avg_Battery_Status": "",
            "Avg_SoC" : "",
            "Avg_Num_Charge" : "",
            "Avg_Vehicle_Capacity" : "",
            "Avg_Customer_Demand" : "",
            "Num_Vehicles": "",
            "Avg_Service_Time" : "",
            "Avg_Customer_TimeWindow" : "",
            "Var_Customer_TimeWindow" : "",
            "Avg_Customer_customer_min_dist" : "",
            "Var_Customer_customer_min_dist" : "",
            "Avg_Customer_station_min_dist" : "",
            "Var_Customer_station_min_dist" : "",
            "Avg_Customer_deposit_dist" : "",
            "Var_Customer_deposit_dist" : "",
            "CounterD_R" : "",
            "CounterD_Rlast" : ""
        }
        ofis = 0
        lastAlgorithmTemp=[100,100,100,100]
        counterD_Rlast=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        linelast=0
        with open('./Counterlast.csv', 'r') as file:
            readerlast = csv.reader(file,delimiter=",")
            for row in readerlast:
                if linelast==0:
                    if row:
                        counterD_Rlast[0]=int(row[0])
                        counterD_Rlast[1]=int(row[1])
                        counterD_Rlast[2]=int(row[2])
                        counterD_Rlast[3]=int(row[3])
                        counterD_Rlast[4]=int(row[4])
                        counterD_Rlast[5]=int(row[5])
                        counterD_Rlast[6]=int(row[6])
                        counterD_Rlast[7]=int(row[7])
                        counterD_Rlast[8]=int(row[8])
                        counterD_Rlast[9]=int(row[9])
                        counterD_Rlast[10]=int(row[10])
                        counterD_Rlast[11]=int(row[11])
                        counterD_Rlast[12]=int(row[12])
                        counterD_Rlast[13]=int(row[13])
                        counterD_Rlast[14]=int(row[14])
                        counterD_Rlast[15]=int(row[15])
                        counterD_Rlast[16]=int(row[16])
                        counterD_Rlast[17]=int(row[17])
                        counterD_Rlast[17]=int(row[18])
                        counterD_Rlast[17]=int(row[19])
                linelast+=1

        # initialize score and probabilities of operators as 1/num_operators
        print("Starting solution OF: ", OF_x_0)
        
        # generate random utilization level for each station
        # compute expected_waiting_time at every station

        starting_time = time.time()
        for iteration in range(1, self.configs["number_iterations"] + 1):
            print("Iteration no", iteration, "of", self.configs["number_iterations"])
            # apply the first stage of the alns
            algorithms_applied = self.apply_destroy_and_repair(
                x_current, iteration
            )
            OF_x_current = x_current.compute_OF()
            # SOLUTION UPDATE
            if x_current.check_solution_feasibility():
                # apply second stage and compute recourse cost
                el_t = time.time()
                recourse_cost = self.apply_second_stage_evaluation(
                    x_current
                )
                print(f"Recourse time: {(time.time()-el_t)*10**6:.2f} \u03BCs ")
                OF_x_current += recourse_cost
                # quit()
                # if there is an improvement, update previous solution
                if OF_x_current < OF_x_previous:
                    x_previous = copy.deepcopy(x_current)
                    OF_x_previous = OF_x_current
                    # assign score to the operators based on "improvement successful"
                    self.update_score_operator(self.operators, self.configs["alns_scores"]["current_better"], algorithms_applied)
                    # if the solution is the best yet update it
                    if OF_x_previous < OF_x_best:
                        x_best = copy.deepcopy(x_current)
                        OF_x_best = OF_x_current
                        # assign score to the operators based on "best solution"
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["global_best"], algorithms_applied)
                else:
                    # apply simulated annealing in order to maybe accept worse solution
                    OF_difference = OF_x_current - OF_x_previous
                    r = np.random.uniform()
                    if r < math.exp((-OF_difference) / (self.configs["annealing_parameters"]["k"] * T)):
                        x_previous = copy.deepcopy(x_current)
                        OF_x_previous = OF_x_current
                        # assign score to the operators based on "solution accepted"
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_accepted"], algorithms_applied)
                        T = T / ( 1 + ( T * self.configs["annealing_parameters"]["frazionamento"] ) )
                    else:
                        # assign score to the operators based on "solution rejected"
                        self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_rejected"], algorithms_applied)
                        x_current = copy.deepcopy(x_previous)
            else:
                # assign score to the operators based on "solution rejected"
                self.update_score_operator(self.operators, self.configs["alns_scores"]["solution_rejected"], algorithms_applied)
                x_current = copy.deepcopy(x_previous)

            # UPDATE WEIGHT OPERATOR CUSTOMER
            if (iteration % self.configs["number_iterations_customer_op_update_weights"]) == 0:
                for i in range(len(self.operators["customer_repair"]["operators"])):
                    self.update_weights(
                        self.operators["customer_repair"]["weight"],
                        i,
                        self.operators["customer_repair"]["score"]
                    )
                
                for i in range(len(self.operators["customer_destroy"]["operators"])):
                    self.update_weights(
                        self.operators["customer_destroy"]["weight"],
                        i,
                        self.operators["customer_destroy"]["score"]
                    )
                self.wipe_scores("customer")

            # UPDATE WEIGHT OPERATOR STATION
            if (iteration % self.configs["number_iterations_station_op_update_weights"]) == 0:

                for i in range(len(self.operators["station_repair"]["operators"])):
                    self.update_weights(
                        self.operators["station_repair"]["weight"],
                        i,
                        self.operators["station_repair"]["score"]
                    )
                for i in range(len(self.operators["station_destroy"]["operators"])):
                    self.update_weights(
                        self.operators["station_destroy"]["weight"],
                        i,
                        self.operators["station_destroy"]["score"]
                    ) 
                
                self.wipe_scores("station")

            if (iteration % self.configs["number_iterations_waiting_update"]) == 0:
                # TODO: completare?
                # compute waiting time parameter alfa, update waiting time
                # if alfa > 1:
                #   apply solution correction (?)
                pass
            print(f"\t {OF_x_current:.2f} {OF_x_best:.2f}")
            if collect_data:
                pass # CODICE RAGAZZI
            
            finishTime = time.time()

            # Inizializzazione DB
            db_Output = open('DB-Output12000_hybrid.csv', 'a', newline='')
            writer = csv.writer(db_Output)
            
            # Instance's Name
            import json
            settings = json.load(open("EVRPTW-main-DBProduction\etc\settings.json"))
            filename = settings["instance_file_name"]
            write_final_row["Instance's Name"] = filename
            write_final_row["Iteration"]=iteration
            # Initial Solution
            write_final_row["Initial Solution"] = x_current.generate_route_id()
            with open(path_soluzione_iniziale, 'w') as file:
                file.write(str(x_current.generate_route_id()))
            # OFIS
            if(iteration == 1):
                write_final_row["OFIS"] = OF_x_0
            else:
                write_final_row["OFIS"] = ofis



            # Moves qui vengono scritte le 4 mosse applicate
            write_final_row["Moves"] = [moves_list["station"]["destroy"][algorithms_applied["station"]["destroy"]["number"]],
                                        moves_list["station"]["repair"][algorithms_applied["station"]["repair"]["number"]],
                                        moves_list["customer"]["destroy"][algorithms_applied["customer"]["destroy"]["number"]],
                                        moves_list["customer"]["repair"][algorithms_applied["customer"]["repair"]["number"]]]
            
            # OFFS
            write_final_row["OFFS"] = OF_x_best
            ofis = OF_x_best
            # OF_Diff
            write_final_row["OF_Diff"] = float(write_final_row["OFIS"]) - float(write_final_row["OFFS"])
            
            # Exe_Time_d-r
            write_final_row["Exe_Time_d-r"] = finishTime-starting_time

            #print(x_current.routes_with_stations)
            '''
            # Avg_Battery_Status
            global_single_nodes_sum = 0
            for route in x_current.routes:
                single_nodes = []
                for node in route:
                    single_nodes.append(node)
                
                for i in range(len(single_nodes)-1):
                    global_single_nodes_sum += self.instance.distance_matrix[single_nodes[i]["name"]][single_nodes[i+1]["name"]]
            
            write_final_row["Avg_Battery_Status"] = global_single_nodes_sum / len(x_current.routes)
            '''
            #-------------------------------------------------------------------------------------
            # Avg_SoC
            soc_sum = 0
            for single_soc in x_current.vehicles:
                soc_sum += single_soc["SoC"]
            
            write_final_row["Avg_SoC"] = soc_sum / len(x_current.vehicles)

            #print(x_current.routes)
            
            # Avg_Num_Charge
            count = 0
            for route in x_current.routes:
                for node in route:
                    if node["name"][0] == 'S':
                        count += 1
            
            write_final_row["Avg_Num_Charge"] = count / len(x_current.vehicles)
            
            # Avg_Vehicle_Capacity
            Current_cargo_sum_c = 0
            for c in x_current.vehicles:
                Current_cargo_sum_c += c["current_cargo"]
            
            write_final_row["Avg_Vehicle_Capacity"] = Current_cargo_sum_c  / len(x_current.vehicles)

            # Avg_Customer_Demand
            sum_tot_demand = 0
            for customer in self.instance.customers:
                sum_tot_demand += customer["demand"]
            
            write_final_row["Avg_Customer_Demand"] = sum_tot_demand / len(self.instance.customers)
            
            # Num_Vehicles
            write_final_row["Num_Vehicles"] = len(x_current.vehicles)

            # Avg_Service_Time
            sum_tot_service_time = 0
            for customer in self.instance.customers:
                sum_tot_service_time += customer["ServiceTime"]
            
            write_final_row["Avg_Service_Time"] = sum_tot_service_time / len(self.instance.customers)

            # Avg_Customer_TimeWindow & Var_Customer_TimeWindow
            sum_tot_customer_timeWindow = []
            for customer in self.instance.customers:
                sum_tot_customer_timeWindow.append(customer["DueDate"] - customer["ReadyTime"])
            
            write_final_row["Avg_Customer_TimeWindow"] = statistics.mean(sum_tot_customer_timeWindow)
            write_final_row["Var_Customer_TimeWindow"] = statistics.variance(sum_tot_customer_timeWindow)
            
            # Avg_Customer_customer_min_dist & Var_Customer_customer_min_dist
            vectordist = []
            for c1 in self.instance.customers:
                temp = 1000000
                for c2 in self.instance.customers:
                    if c1 != c2:
                        dist=math.sqrt(pow((c1['x']-c2['x']),2)+pow((c1['y']-c2['y']),2))
                        if  dist<temp:
                            temp=dist
                vectordist.append(temp)
            write_final_row["Avg_Customer_customer_min_dist"] = statistics.mean(vectordist)
            write_final_row["Var_Customer_customer_min_dist"] = statistics.variance(vectordist)

            # Avg_Customer_station_min_dist & Var_Customer_station_min_dist
            vectordiststation = []
            for c1 in self.instance.customers:
                temp = 1000000
                for s1 in self.instance.charging_stations:
                    if c1 != s1:
                        dist=math.sqrt(pow((c1['x']-s1['x']),2)+pow((c1['y']-s1['y']),2))
                        if  dist<temp:
                            temp=dist
                vectordiststation.append(temp)
            write_final_row["Avg_Customer_station_min_dist"] = statistics.mean(vectordiststation)
            write_final_row["Var_Customer_station_min_dist"] = statistics.variance(vectordiststation)
            # Avg_Customer_deposit_dist & Var_Customer_deposit_dist
            vectordistdeposit = []
            for c1 in self.instance.customers:
                dist=(math.sqrt(pow((c1['x']-self.instance.depot_point['x']),2)+pow((c1['y']-self.instance.depot_point['y']),2)))
                vectordistdeposit.append(dist)
            write_final_row["Avg_Customer_deposit_dist"] = statistics.mean(vectordistdeposit)
            write_final_row["Var_Customer_deposit_dist"] = statistics.variance(vectordistdeposit)
            
            #-------------------------------------------------------------------------------------
            
            line=0
            counterD_R=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            import ast

            with open('./Counter.csv', 'r') as file:
                reader = csv.reader(file,delimiter=",")
                for row in reader:
                    if line==0:
                        if row:
                            counterD_R[0]=int(row[0])
                            counterD_R[1]=int(row[1])
                            counterD_R[2]=int(row[2])
                            counterD_R[3]=int(row[3])
                            counterD_R[4]=int(row[4])
                            counterD_R[5]=int(row[5])
                            counterD_R[6]=int(row[6])
                            counterD_R[7]=int(row[7])
                            counterD_R[8]=int(row[8])
                            counterD_R[9]=int(row[9])
                            counterD_R[10]=int(row[10])
                            counterD_R[11]=int(row[11])
                            counterD_R[12]=int(row[12])
                            counterD_R[13]=int(row[13])
                            counterD_R[14]=int(row[14])
                            counterD_R[15]=int(row[15])
                            counterD_R[16]=int(row[16])
                            counterD_R[17]=int(row[17])
                    line+=1

            if(write_final_row["OF_Diff"])>0:
                if algorithms_applied["station"]["destroy"]["number"]==0: 
                    counterD_R[0]+=1
                    lastAlgorithmTemp[0]=0
                if algorithms_applied["station"]["destroy"]["number"]==1: 
                    counterD_R[1]+=1
                    lastAlgorithmTemp[0]=1
                #--------------------------------------------------------------
                if algorithms_applied["station"]["repair"]["number"]==0: 
                    counterD_R[2]+=1
                    lastAlgorithmTemp[1]=0
                if algorithms_applied["station"]["repair"]["number"]==1: 
                    counterD_R[3]+=1
                    lastAlgorithmTemp[1]=1
                #--------------------------------------------------------------
                if algorithms_applied["customer"]["destroy"]["number"]==0: 
                    counterD_R[4]+=1
                    lastAlgorithmTemp[2]=0
                if algorithms_applied["customer"]["destroy"]["number"]==1: 
                    counterD_R[5]+=1
                    lastAlgorithmTemp[2]=1
                if algorithms_applied["customer"]["destroy"]["number"]==2: 
                    counterD_R[6]+=1
                    lastAlgorithmTemp[2]=2
                if algorithms_applied["customer"]["destroy"]["number"]==3: 
                    counterD_R[7]+=1
                    lastAlgorithmTemp[2]=3
                if algorithms_applied["customer"]["destroy"]["number"]==4: 
                    counterD_R[8]+=1
                    lastAlgorithmTemp[2]=4
                if algorithms_applied["customer"]["destroy"]["number"]==5: 
                    counterD_R[9]+=1
                    lastAlgorithmTemp[2]=5
                if algorithms_applied["customer"]["destroy"]["number"]==6: 
                    counterD_R[10]+=1
                    lastAlgorithmTemp[2]=6
                if algorithms_applied["customer"]["destroy"]["number"]==7: 
                    counterD_R[11]+=1
                    lastAlgorithmTemp[2]=7
                if algorithms_applied["customer"]["destroy"]["number"]==8: 
                    counterD_R[12]+=1
                    lastAlgorithmTemp[2]=8
                if algorithms_applied["customer"]["destroy"]["number"]==9: 
                    counterD_R[13]+=1
                    lastAlgorithmTemp[2]=9
                if algorithms_applied["customer"]["destroy"]["number"]==10: 
                    counterD_R[14]+=1
                    lastAlgorithmTemp[2]=10
                #--------------------------------------------------------------
                if algorithms_applied["customer"]["repair"]["number"]==0: 
                    counterD_R[15]+=1
                    lastAlgorithmTemp[3]=0
                if algorithms_applied["customer"]["repair"]["number"]==1: 
                    counterD_R[16]+= 1
                    lastAlgorithmTemp[3]=1
                if algorithms_applied["customer"]["repair"]["number"]==2: 
                    counterD_R[17]+=1
                    lastAlgorithmTemp[3]=2

            write_final_row["CounterD_R"]= counterD_R
            write_final_row["CounterD_Rlast"]= counterD_Rlast

            f = open('./Counter.csv', 'w')
            writerf = csv.writer(f)
            writerf.writerow(counterD_R)
            f.close()
          
            # FINAL PRINT
            writer.writerow([write_final_row["Instance's Name"],write_final_row["Iteration"],ran_seed, write_final_row["Initial Solution"],write_final_row["OFIS"],write_final_row["Moves"],write_final_row["OFFS"],
                             write_final_row["OF_Diff"],write_final_row["Exe_Time_d-r"],write_final_row["Avg_Battery_Status"],write_final_row["Avg_SoC"],write_final_row["Avg_Num_Charge"],write_final_row["Avg_Vehicle_Capacity"],
                             write_final_row["Avg_Customer_Demand"],write_final_row["Num_Vehicles"],write_final_row["Avg_Service_Time"],write_final_row["Avg_Customer_TimeWindow"],write_final_row["Var_Customer_TimeWindow"],
                             write_final_row["Avg_Customer_customer_min_dist"],write_final_row["Var_Customer_customer_min_dist"],write_final_row["Avg_Customer_station_min_dist"],
                             write_final_row["Var_Customer_station_min_dist"],write_final_row["Avg_Customer_deposit_dist"], write_final_row["Var_Customer_deposit_dist"], write_final_row["CounterD_R"], write_final_row["CounterD_Rlast"]])
            db_Output.close()
            
        if lastAlgorithmTemp[0]==0:
            counterD_Rlast[0]+=1
        if lastAlgorithmTemp[0]==1:
            counterD_Rlast[1]+=1

        if lastAlgorithmTemp[1]==0:
            counterD_Rlast[2]+=1
        if lastAlgorithmTemp[1]==1:
            counterD_Rlast[3]+=1

        if lastAlgorithmTemp[2]==0:
            counterD_Rlast[4]+=1
        if lastAlgorithmTemp[2]==1:
            counterD_Rlast[5]+=1
        if lastAlgorithmTemp[2]==2:
            counterD_Rlast[6]+=1
        if lastAlgorithmTemp[2]==3:
            counterD_Rlast[7]+=1
        if lastAlgorithmTemp[2]==4:
            counterD_Rlast[8]+=1
        if lastAlgorithmTemp[2]==5:
            counterD_Rlast[9]+=1
        if lastAlgorithmTemp[2]==6:
            counterD_Rlast[10]+=1
        if lastAlgorithmTemp[2]==7:
            counterD_Rlast[11]+=1
        if lastAlgorithmTemp[2]==8:
            counterD_Rlast[12]+=1
        if lastAlgorithmTemp[2]==9:
            counterD_Rlast[13]+=1
        if lastAlgorithmTemp[2]==10:
            counterD_Rlast[14]+=1

        if lastAlgorithmTemp[3]==0:
            counterD_Rlast[15]+=1
        if lastAlgorithmTemp[3]==1:
            counterD_Rlast[16]+=1
        if lastAlgorithmTemp[3]==2:
            counterD_Rlast[17]+=1
        
        f2 = open('./Counterlast.csv', 'w')
        writerf2 = csv.writer(f2)
        writerf2.writerow(counterD_Rlast)
        f2.close()
        
        print("Starting solution OF: ", OF_x_0)
        print("Final solution OF: ", OF_x_best)
        print("Total elapsed time: ", time.time() - starting_time)
        plotter = Plotter(self.instance, x_best)
        #plotter.show_solution()
        return x_best, OF_x_best
    #aggiorna lo score iniziale che è 0 aggiungendo which_score (sempre settato a 1 in EVRPTW-main-DBProduction\etc\settings_solver.json)
    def update_score_operator(self, operators, which_score, algorithms_applied):
        # update score of the operator based on the value contained in which_score
        if algorithms_applied["station"]["repair"]["number"] != 999:
            operators["station_destroy"]["score"][algorithms_applied["station"]["destroy"]["number"]] += which_score
            operators["station_repair"]["score"][algorithms_applied["station"]["repair"]["number"]] += which_score
        else:
            operators["customer_destroy"]["score"][algorithms_applied["customer"]["destroy"]["number"]] += which_score
            operators["customer_repair"]["score"][algorithms_applied["customer"]["repair"]["number"]] += which_score

    def update_weights(self, operator, apply_on, score):
        # apply update formula score = (score * lambda) + (1 - lambda) * score where lambda is the decay parameter
        operator[apply_on] = (operator[apply_on] * self.configs["alns_decay_parameter"]) + ( 1 - self.configs["alns_decay_parameter"]) * score[apply_on]

    def wipe_scores(self, which):
        # reset scores after weight update
        for i in range(len(self.operators[which + "_destroy"]["operators"])):
            self.operators[which + "_destroy"]["score"][i] = 0
        for i in range(len(self.operators[which + "_repair"]["operators"])):
            self.operators[which + "_repair"]["score"][i] = 0

    def _select_method_0(self, op_list, weights):
        # select operator to be applied
        pos = random.choices(
            range(len(op_list)), #restituisce una sequenza di numeri che vanno da 0 a len(op_list) - 1
            weights
        )[0]
        return op_list[pos], pos 
    
    #random pesato delle migliori (>0.5)
    def _select_method_1(self, op_list, weights):
        with open(path_soluzione_iniziale, 'r') as file:
            content = file.read()
        #print(op_list)
        init_sol=clean(content)
        dir_models="EVRPTW-main-DBProduction\models\Convolutional\models_10fold_DEF\\12000_Train\_15ep_32bs"
        models=os.listdir(dir_models)  
        for i in range(len(models)):
            models[i] = models[i].replace(".csv.h5", "") 
        predictions={}
        for i in range(len(op_list)):

            pos=i
            # select operator to be applied
            #pos = random.choices(range(len(op_list)),weights)[0]
            move = op_list[i].__class__.__name__
            if move in models:
                #print("Models here")
                m_file=dir_models+"\\"+move+".csv.h5"
                model = keras.models.load_model(m_file)
            y_pred = model.predict(init_sol.reshape(1, -1))
            predictions[pos]=y_pred[0][0]
        print(predictions)
        #randomizzazione pesata

        migliorative = {k: v for k, v in predictions.items() if v > 0.5}
        if len(migliorative)>0:
            chiavi = list(migliorative.keys())
            valori = list(migliorative.values())
            chiave_scelta = random.choices(chiavi, weights=valori, k=1)[0]


            best=op_list[chiave_scelta]
            pos=chiave_scelta
            #print(best)
            #print(chiave_scelta)
        else:pos = random.choices(range(len(op_list)),weights)[0]; best=op_list[pos]
        if best==None:  pos = random.choices(range(len(op_list)),weights)[0]; best=op_list[pos]
        return best, pos

    def apply_destroy_and_repair(self, x: Solution, iteration: int):
        # apply the first stage of the alns
        couples = {
            "station" : {
                "destroy" : {
                    "number" : 999,
                    "time" : 0
                },
                "repair" : {
                    "number" : 999,
                    "time" : 0
                }
            },
            "customer" : {
                "destroy" : {
                    "number" : 0,
                    "time" : 0 
                },
                "repair" : {
                    "number" : 0,
                    "time" : 0
                },
            }
        }
        if (iteration % self.configs["number_iterations_station_removal"]) == 0:
            if iteration==2000 or iteration==3000 or iteration==4000 or iteration==5000 or iteration==6000 or iteration==7000 or iteration==8000 or iteration==9000 or iteration==10000 or iteration==11000 or iteration==12000:
                # SELECT DESTROY STATION
                station_destroy, pos_destroy = self._select_method_1(
                    self.operators["station_destroy"]["operators"],
                    self.operators["station_destroy"]["weight"]
                )

                # SELECT REPAIR STATION
                station_repair, pos_repair = self._select_method_1(
                    self.operators["station_repair"]["operators"],
                    self.operators["station_destroy"]["weight"]
                )
            else: 
                # SELECT DESTROY STATION
                station_destroy, pos_destroy = self._select_method_0(
                    self.operators["station_destroy"]["operators"],
                    self.operators["station_destroy"]["weight"]
                )

                # SELECT REPAIR STATION
                station_repair, pos_repair = self._select_method_0(
                    self.operators["station_repair"]["operators"],
                    self.operators["station_destroy"]["weight"]
                )

            # APPLY DESTROY STATION
            currtime = time.time()
            removed_stations = station_destroy.apply(x)
            time_elapsed_destroy = time.time() - currtime
            couples["station"]["destroy"]["number"] = pos_destroy
            couples["station"]["destroy"]["time"] = time_elapsed_destroy
            print(f'Applying STATION destroy [{station_destroy}] {time_elapsed_destroy*10**6:.2f} \u03BCs')
            
            # APPLY REPAIR STATION
            currtime = time.time()
            station_repair.apply(x, removed_stations)
            time_elapsed_repair = time.time() - currtime
            couples["station"]["repair"]["number"] = pos_repair
            couples["station"]["repair"]["time"] = time_elapsed_repair

            print(f'Applying STATION repair [{station_repair}] {time_elapsed_repair*10**6:.2f} \u03BCs')

        # SELECT DESTROY CUSTOMER
        if iteration==2000 or iteration==3000 or iteration==4000 or iteration==5000 or iteration==6000 or iteration==7000 or iteration==8000 or iteration==9000 or iteration==10000 or iteration==11000 or iteration==12000:

            customer_destroy, pos_destroy = self._select_method_1(
                self.operators["customer_destroy"]["operators"], 
                self.operators["customer_destroy"]["weight"]
            )
            # SELECT REPAIR CUSTOMER
            customer_repair, pos_repair = self._select_method_1(
                self.operators["customer_repair"]["operators"],
                self.operators["customer_repair"]["weight"]
            )
        else:
            customer_destroy, pos_destroy = self._select_method_0(
                self.operators["customer_destroy"]["operators"], 
                self.operators["customer_destroy"]["weight"]
            )
            # SELECT REPAIR CUSTOMER
            customer_repair, pos_repair = self._select_method_0(
                self.operators["customer_repair"]["operators"],
                self.operators["customer_repair"]["weight"]
            )
        # APPLY REMOVE CUSTOMER
        # customer_destroy = self.operators["customer_destroy"]["operators"][10]
        currtime = time.time()
        removed_customers = customer_destroy.apply(x)
        time_elapsed_destroy = time.time() - currtime
        print(f'Applying CUSTOMER destroy [{customer_destroy}] {time_elapsed_destroy*10**6:.2f} \u03BCs')
        couples["customer"]["destroy"]["number"] = pos_destroy
        couples["customer"]["destroy"]["time"] = time_elapsed_destroy

        # TEST CUSTOMER REPAIR
        # 1 ProbabilisticGreedyRepairCustomer(operators_configs),
        # 2 ProbabilisticGreedyConfidenceRepairCustomer(operators_configs),
        customer_repair = self.operators["customer_repair"]["operators"][0]
        # currtime = time.time()
        # customer_repair.apply(x, removed_customers)
        # time_elapsed_destroy = time.time() - currtime
        # print(f'>> Applying STATION destroy [{customer_repair} {time_elapsed_destroy*10**6:.2f}] \u03BCs')
        # quit()
        
        # APPLY REPAIR CUSTOMER
        currtime = time.time()
        customer_repair.apply(x, removed_customers)
        time_elapsed_repair = time.time() - currtime
        couples["customer"]["repair"]["number"] = pos_repair
        couples["customer"]["repair"]["time"] = time_elapsed_repair
        print(f'Applying CUSTOMER repair [{customer_repair}] {time_elapsed_repair*10**6:.2f} \u03BCs')
        # print(f'Applying CUSTOMER destroy [{customer_destroy} {time_elapsed_destroy:.2f}] and repair [{customer_repair} {time_elapsed_repair:.2f}]')
        x.remove_empty_routes()

        return couples

    def apply_second_stage_evaluation(self, solution: Solution):
        # apply the second stage and compute the average recourse cost
        E_cost_k = np.zeros(len(solution.routes))
        for route_idx in range(len(solution.routes)):
            E_cost_k[route_idx] = self.evaluate_second_stage(solution, route_idx)   
        return np.mean(E_cost_k)
    
    # @profile
    def evaluate_second_stage(self, first_stage_solution: Solution,  route_idx: int):
        first_stage_solution.clean_solution(route_idx)
        # generate the position of the stations
        first_stage_solution.generate_charging_station_pos_route(route_idx)
        old_waiting_time = self.instance.E_recharge
        # expected cost
        E_cost_k = 0
        route = first_stage_solution.routes[route_idx]
        # cost per scenario
        cost_k = 0
        # enter only if there is a station in the route
        for station_index in first_stage_solution.charging_station_pos[route_idx]:
            route_energy = self.instance.compute_energy_cost(route, node_nmb = station_index)
            # get station arrival (NB: station_index -1 because the depot has no arrival time)
            station_arrival = first_stage_solution.arrival_times[route_idx][station_index - 1]
            # get customer after station and remove the last element (the depot)
            nodes_after_station = first_stage_solution.routes[route_idx][station_index+1:-1]
            customer_due_date = []
            for el, customer in enumerate(nodes_after_station):
                if customer["isCustomer"]:
                    # add due date if the node is a customer
                    customer_due_date.append(
                        self.instance.customers_dict[customer["name"]]["DueDate"]
                    )
            customer_due_date = np.array(customer_due_date)
            
            station = self.instance.stations_dict[first_stage_solution.routes[route_idx][station_index]["name"]]
            for _ in range(self.configs["n_scenarios"]):
                # Compute new waiting times
                # change this to only 1 
                local_arrival_times = []
                # check if the station is busy or free
                is_free = random.random() < station["utilization_level"]
                if is_free:
                    new_waiting_time = 0
                    tmp_old_waiting_time = 0
                else:
                    new_waiting_time = np.random.exponential(1 / first_stage_solution.instance.mu)
                    tmp_old_waiting_time = old_waiting_time
                    # update the arrival times of customer after recharging station
                for i in range(station_index, len(first_stage_solution.routes[route_idx]) - 1):
                    # (NB: station_index -1 because the depot has no arrival time)
                    local_arrival_times.append(
                        first_stage_solution.arrival_times[route_idx][i] + tmp_old_waiting_time - new_waiting_time
                    )
                
                # get customer after station and remove the last element (the depot)
                arrival_times_after_station = local_arrival_times[:-1]
                # get customer due date
                try:
                    for el, customer in enumerate(nodes_after_station):
                        if customer["isStation"]:
                            # if the node is a station remove the corresponding arrival_times
                            arrival_times_after_station.pop(el)


                    arrival_times_after_station = np.array(arrival_times_after_station)
                    # find the violated due dates (i.e. arrival_times_after_station > customer_due_date)
                    violated = np.maximum(arrival_times_after_station - customer_due_date, np.zeros(len(customer_due_date)))
                except:
                    violated = []

                for violated_idx, violation in enumerate(violated):
                    # if the is a customer,
                    if violation != 0 and nodes_after_station[violated_idx]["isCustomer"]:
                        
                        success_recharge_partial = False
                        success_pickup_exchange = False
                        # try partial recharge
                        if self.consider_partial_recharge:
                            cost_recourse, success_recharge_partial = self.partial_recharge_recourse(
                                first_stage_solution,
                                route_idx,
                                route_energy,
                                nodes_after_station[violated_idx],
                                node_nmb = station_index
                            )
                            # if successful, save cost
                            if success_recharge_partial:
                                cost_k += cost_recourse
                            
                        # try exchange pickup
                        if self.consider_pickup and (not success_recharge_partial):
                            cost_recourse, success_pickup_exchange = self.pickup_exchange_recourse(
                                first_stage_solution,
                                nodes_after_station[violated_idx],
                                station_arrival,
                                route_idx,
                                route_energy,
                                node_nmb = station_index
                            )
                            # if successful save cost
                            if success_pickup_exchange:
                                cost_k += cost_recourse
                        # if both partial recharge and exchange pickup failed, add a new route
                        if (not success_recharge_partial) and (not success_pickup_exchange):
                            # adding a new route in first_stage_solution
                            cost_k += self.new_route_recourse(
                                first_stage_solution,
                                nodes_after_station[violated_idx],
                                station_arrival,
                                route_idx,
                                route_energy,
                                node_nmb = station_index
                            )                

        # aggiungo il valore atteso di costo di ogni route
        
        E_cost_k = cost_k / self.configs["n_scenarios"]
        return E_cost_k

    # @ profile
    def pickup_exchange_recourse(self, solution : Solution, customer_violated : dict, station_arrival : float, route_idx, route_energy, node_nmb) -> float:
        # get time and idx of the elements at the moment the recourse vehicle arrives at the station
        route_times = {}
        arrival_nd = [np.array(route) for route in solution.arrival_times]
        for i, route in enumerate(arrival_nd):
            if i != route_idx:
                # for each route save the next customer and the arrival time
                node_idx = np.argmax(route > station_arrival)
                if node_idx + 1 >= len(route):
                    route_times.update({
                        (i, node_idx + 1) : route[node_idx] - station_arrival
                    })

        # sort it in increasing order (lowest time difference first)
        sorted_times = dict(sorted(route_times.items(), key = lambda item: item[1]))
        route_position = list(sorted_times.keys())
        # try to add the customer_violated customer to each route and see if route is still feasible
        
        for couple in route_position:
            route, position = couple[0], couple[1]
            

            # compute the energy of the old route
            test_route_energy = self.instance.compute_energy_cost(solution.routes[route], position + 1)


            distance_list = []
            if solution.routes[route][position]["isDepot"]:
                route_feasible = False
            else:
                # compute new distance of the tested route, check for feasibility as well
                route_feasible, distance_new_node, distance_list = self.node_to_new_route_test(solution, position, route, customer_violated)
            # if the route is still feasible, compute recourse cost otherwise go to next route
            if route_feasible:
                # get new end time of the route which we are trying to add the node to
                t_new = self.compute_end_time(
                    solution, 
                    route_idx, 
                    distance_list, 
                    position, 
                    (self.instance.vehicle_consumption_rate * distance_new_node) / self.instance.battery_tank_capacity,
                    customer_violated,
                    is_pickup_exchange = True
                    )
                # compute overtime for the route with the new node
                overtime_new = max(t_new - solution.instance.depot_point["DueDate"], 0)
                # compute distance of the old route without the violated customer
                total_distance = 0
                distance_list = []
                for j in range(len(solution.routes[route_idx]) - 1, node_nmb, -1):
                    total_distance += self.instance.distance_matrix._get_item(
                            solution.routes[route_idx][j]["name"],
                            solution.routes[route_idx][j-1]["name"]
                        )
                    distance_list.append(self.instance.distance_matrix._get_item(
                            solution.routes[route_idx][j]["name"],
                            solution.routes[route_idx][j-1]["name"]
                        ))
                SoC_consumed_perc = (self.instance.vehicle_consumption_rate * total_distance) / self.instance.battery_tank_capacity 
                # compute final time for the route which had the violated customer
                t = self.compute_end_time(solution, route_idx, distance_list, node_nmb, SoC_consumed_perc, customer_violated, is_new_route = True)
                overtime = max(t - solution.instance.depot_point["DueDate"], 0)
                



                energy_difference = route_energy - self.instance.compute_energy_cost(
                    solution.routes[route_idx], node_nmb
                )
                test_energy_difference = test_route_energy - distance_new_node * self.instance.cost_energy
                # update the cost k compute the OF with the new values for time, overtime and distance
                cost_k = (total_distance*solution.instance.cost_energy + t*solution.instance.cost_driver + overtime*solution.instance.cost_overtime + solution.instance.cost_vehicle) - (
                    self.instance.cost_energy * energy_difference
                ) + (
                    distance_new_node*solution.instance.cost_energy + t_new*solution.instance.cost_driver + overtime_new*solution.instance.cost_overtime + solution.instance.cost_vehicle
                ) - (self.instance.cost_energy * test_energy_difference)
                
                # quit()
                return cost_k, True

                

        # quit()
        return 0, False

    def node_to_new_route_test(self, solution : Solution, position_idx : int, new_route_idx : int, customer_violated : dict):
        # compute the distance from node at position_idx passing through customer violated to depot
        distance_list = [
            self.instance.distance_matrix._get_item(
                solution.routes[new_route_idx][position_idx]["name"],
                customer_violated["name"]),
            self.instance.distance_matrix._get_item(
                customer_violated["name"],
                solution.routes[new_route_idx][position_idx+1]["name"]
        )
        ]
        # starting distance from position_idx to position_idx + 1 passing through the violated customer
        total_distance = sum(distance_list) 

        # total distance of the test route
        for j in range(position_idx+1, len(solution.routes[new_route_idx]) - 1):
            dist_element = self.instance.distance_matrix._get_item(
                solution.routes[new_route_idx][j]["name"],
                solution.routes[new_route_idx][j+1]["name"]
            )
            total_distance += dist_element
            distance_list.append(dist_element)

        # compute soc consumed and check for feasibility of the application
        SoC_consumed_perc = (self.instance.vehicle_consumption_rate * total_distance) / self.instance.battery_tank_capacity 
        if SoC_consumed_perc > 0.9:
            return False, 0, []
        else:
            return True, total_distance, distance_list

    
    def new_route_recourse(self, solution : Solution, violated : dict, station_arrival : float, route_idx, route_energy, node_nmb) -> float:
        total_distance = 0
        distance_list = []
        for j in range(len(solution.routes[route_idx]) - 1, node_nmb, -1):
            total_distance += self.instance.distance_matrix._get_item(
                    solution.routes[route_idx][j]["name"],
                    solution.routes[route_idx][j-1]["name"]
                )
            distance_list.append(self.instance.distance_matrix._get_item(
                    solution.routes[route_idx][j]["name"],
                    solution.routes[route_idx][j-1]["name"]
                ))
        SoC_consumed_perc = (self.instance.vehicle_consumption_rate * total_distance) / self.instance.battery_tank_capacity 
        t = self.compute_end_time(solution, route_idx, distance_list, node_nmb, SoC_consumed_perc, violated, is_new_route = True)
        overtime = max(t - solution.instance.depot_point["DueDate"], 0)
        

        tmp_solution = Solution(self.instance)
        # creo route e aggiungo nodo e depot
        tmp_solution.add_new_route(station_arrival)
        tmp_solution.add_node(
            node = violated["name"], 
            route_idx = 0,
            position_idx = 1
        )
        tmp_solution.add_node(
            node = self.instance.depot_point["StringID"],
            route_idx = 0,
            position_idx = 2
        )
        # calcolo l'energia risparmiata con questa operazione
        energy_difference = route_energy - self.instance.compute_energy_cost(
            solution.routes[route_idx], node_nmb
        )
        #change
        cost_k = (total_distance*solution.instance.cost_energy + t*solution.instance.cost_driver + overtime*solution.instance.cost_overtime + solution.instance.cost_vehicle) - (
            self.instance.cost_energy * energy_difference
        ) + tmp_solution.compute_OF_route(
            0
        )

        if tmp_solution.vehicles[0]["time"] > self.instance.depot_point["DueDate"]:
            cost_k += self.instance.cost_driver*(self.instance.depot_point["DueDate"] - tmp_solution.vehicles[0]["time"])
            cost_k += self.instance.cost_overtime*(tmp_solution.vehicles[0]["time"] - self.instance.depot_point["DueDate"])
        else:
            cost_k += self.instance.cost_driver * (tmp_solution.vehicles[0]["time"] - self.instance.depot_point["DueDate"])
        
        return cost_k

    def partial_recharge_recourse(self, solution: Solution, route_idx, route_energy, violated, node_nmb = 0):
        feasible = True
        
        # for each station index
        for station_idx in solution.charging_station_pos[route_idx]:
            total_distance = 0
            # increasing distance between each node
            distance_list = []
            # compute total distance from station to depot
            for j in range(len(solution.routes[route_idx]) - 1, station_idx, -1):
                total_distance += self.instance.distance_matrix._get_item(
                    solution.routes[route_idx][j]["name"],
                    solution.routes[route_idx][j-1]["name"]
                )
                distance_list.append(self.instance.distance_matrix._get_item(
                    solution.routes[route_idx][j]["name"],
                    solution.routes[route_idx][j-1]["name"]
                ))


            SoC_consumed_perc = (self.instance.vehicle_consumption_rate * total_distance) / self.instance.battery_tank_capacity 
            # soc_tolerance: min SoC to arrive in the depot
            target_SoC = SoC_consumed_perc + self.configs['SoC_tolerance']
            # if target_soc goes over 0.9, it's set to 0.9 -> charge at
            if target_SoC > 0.9:
                target_SoC = 0.9
                feasible = False
            
            # update just the vehicle time
            t = self.compute_end_time(solution, route_idx, distance_list, station_idx, SoC_consumed_perc, violated)
            overtime = max(t - solution.instance.depot_point["DueDate"], 0)
            

        if feasible:
            energy_difference = route_energy - solution.instance.compute_energy_cost(
                solution.routes[route_idx], node_nmb = node_nmb
            )
            cost_k = (
                total_distance*solution.instance.cost_energy + t*solution.instance.cost_driver + overtime*solution.instance.cost_overtime + solution.instance.cost_vehicle
                ) - (self.instance.cost_energy * energy_difference)

            if t > self.instance.depot_point["DueDate"]:
                cost_k += self.instance.cost_driver*(self.instance.depot_point["DueDate"] - t)
                cost_k += self.instance.cost_overtime*(t - self.instance.depot_point["DueDate"])
            else:
                cost_k += self.instance.cost_driver * (t - self.instance.depot_point["DueDate"])
        
            return cost_k, feasible
        
        else:
            return 0, feasible

    def compute_end_time(self, solution : Solution, route_idx, distance_list, start_point_idx, SoC_consumed_perc, violated, is_pickup_exchange = False, is_new_route = False):
        t = solution.arrival_times[route_idx][start_point_idx - 1]
        # index to keep track of distances
        i = len(distance_list) - 1
        for node_idx in range(len(solution.routes[route_idx]) - 1, start_point_idx, -1):
            if i < 0:
                break
            if solution.routes[route_idx][node_idx]["name"] == violated["name"] and is_new_route:
                pass
            else:
                if solution.routes[route_idx][node_idx]["isCustomer"]:
                    customer = solution.instance.customers_dict[solution.routes[route_idx][node_idx]["name"]]
                    current_time = t + distance_list[i] / solution.instance.average_velocity
                    i -= 1
                    if t < customer["ReadyTime"]:
                        t = customer["ReadyTime"] + customer["ServiceTime"]
                    else:
                        t = current_time + customer["ServiceTime"]
                elif solution.routes[route_idx][node_idx]["isStation"]:
                    expected_waiting_time = solution.instance.get_expected_waiting_time(
                        solution.instance.stations_dict[solution.routes[route_idx][node_idx]["name"]]["utilization_level"]
                    )
                    current_time = distance_list[i] / solution.instance.average_velocity
                    i -= 1
                    t += current_time + expected_waiting_time + (solution.instance.E_recharge * (SoC_consumed_perc - self.configs['SoC_tolerance']))
                else:
                    current_time = distance_list[i] / solution.instance.average_velocity
                    i -= 1
                    t += current_time

        return t

    def check_next_customer_infeasible(self, solution: Solution, customer_pos: list, station_delays, station_idx):
        # keskin simulation algorithm for next customer infeasibility
        # take number of scenarios
        n_scenarios = self.configs["n_scenarios"]
        route_idx = customer_pos[0]
        node_idx = customer_pos[1]
        prob = 0
        # take arrival time at the station (-1 because arrival_times excludes the time at the first depot), i.e. t_0
        station_arrival = np.ones(n_scenarios) * solution.arrival_times[route_idx][station_idx - 1]
        # extract node i from solution
        node = solution.routes[route_idx][node_idx]
        if not node["isCustomer"]:
            prob = 0
        else:
            # take late window of i
            end_time_window_node = solution.instance.customers_dict[node["name"]]["DueDate"]
            # take service time of i
            service_time_node = solution.instance.customers_dict[node["name"]]["ServiceTime"]
            # take distance iteratively from station to i
            distance = 0
            for node_nmb in range(len(solution.routes[route_idx][station_idx : node_idx])):
                distance += solution.instance.distance_matrix._get_item(
                    solution.routes[route_idx][station_idx + node_nmb]["name"], 
                    solution.routes[route_idx][station_idx + node_nmb + 1]["name"]
                )
            # compute the time needed to pass that distance
            time_distance = solution.instance.average_velocity * distance

            # compute probability that customer i + 1 is infeasible
            time_to_pickup_array = np.ones(n_scenarios) * (end_time_window_node - service_time_node - time_distance)

            prob = sum((station_arrival + station_delays) > time_to_pickup_array) / n_scenarios

        return prob

    def check_route_feasible_after_insertion(self, solution : Solution, route_idx):
        # keskin simulation algorithm for toute feasible after insertion
        it = 1
        prob = 0
        n_scenarios = self.configs["n_scenarios"]
        E_service_time = self.instance.E_recharge

        while it <= n_scenarios:
            base_solution = copy.deepcopy(solution)
            base_solution.update_station_waiting_time(route_idx, self.instance.E_recharge)
            base_solution.generate_charging_station_pos_route(route_idx)

            for i, node in enumerate(base_solution.routes[route_idx]):
                for station_index in base_solution.charging_station_pos[route_idx]:
                    if i > station_index and not node["isDepot"]:
                        customer_time = self.instance.customers_dict[node["name"]]["DueDate"]
                        if base_solution.arrival_times[route_idx][i-1] >=  customer_time:
                            prob += 1
                            break
            it += 1
        prob = 1 - prob / n_scenarios
        return prob

    def construct_initial_solution(self) -> Solution:

        """Solves the first stage solution.
        Steps are as follows:
        1. Get all customers in a stack, take one at a time and add them to a route. If an EV is near its due depot time
          it looks for a station if it needs charge, otherwise gets back to a depot and the next EV starts its route.
        2. If en route the EV needs charge, instead of going to a customer it looks for an available recharge station,
          recharging 40% of the charge (as mentioned in Keskin) then restarts. The recharging stations have a random
          waiting time, that gets realized when the ev reaches the station.
        3. Then the ev waits how much it has to wait and restarts its route after waiting and recharging.


        Returns:
            (routes, vehicles): the solution to the first step is returned as a dict, in [0] the list of routes used and in [1]
                the list of dictionaries of vehicles that have served, with a field that represents the list of times at which it reached a point
                in the route
        """

        run = True
        customer_name_lst = list(self.instance.customers_dict.keys())
        
        solution = Solution(self.instance)
        route_nmb = 0
        node_index = 1
        solution.add_new_route()
        
        while len(customer_name_lst) != 0:
            customer_to_add = self.get_nearest_customer(
                solution.routes[route_nmb][-1]["name"],
                solution.vehicles[route_nmb],
                customer_name_lst
            )
            if solution.routes[route_nmb][node_index - 1]["isDepot"]:
                solution.add_node(customer_to_add, route_nmb, node_index)
                customer_name_lst.remove(customer_to_add)
                node_index += 1
            else:
                # if no pickup can be done
                if customer_to_add != "D0":
                    # compute distance to travel
                    potential_distance_travelled = self.instance.distance_matrix._get_item(
                        solution.routes[route_nmb][-1]["name"],
                        customer_to_add
                        )
                    # compute state of charge
                    new_soc = (solution.vehicles[route_nmb]["SoC"]*self.instance.battery_tank_capacity - self.instance.vehicle_consumption_rate * potential_distance_travelled) / self.instance.battery_tank_capacity
                    # if not feasible
                    if new_soc < 0.1:
                        # add station
                        station_to_add = self.get_nearest_station(solution.routes[route_nmb][-1]["name"])
                        solution.add_node(station_to_add, route_nmb, node_index)
                        node_index += 1
                    else:
                        # add the customer to visit
                        solution.add_node(customer_to_add, route_nmb, node_index)
                        customer_name_lst.remove(customer_to_add)
                        node_index += 1
                else:
                    # customer_to_add is the depot ("D0")
                    potential_distance_travelled = self.instance.distance_matrix._get_item(
                        solution.routes[route_nmb][-1]["name"],
                        customer_to_add
                    )
                    new_soc = (solution.vehicles[route_nmb]["SoC"]*self.instance.battery_tank_capacity - self.instance.vehicle_consumption_rate * potential_distance_travelled) / self.instance.battery_tank_capacity
                    # if needed recharge to arrive to the depot
                    if new_soc < 0.1:
                        station_to_add = self.get_nearest_station(solution.routes[route_nmb][-1]["name"])
                        solution.add_node(station_to_add, route_nmb, node_index)
                        node_index += 1
                    else:
                        solution.add_node(customer_to_add, route_nmb, node_index)

                        route_nmb += 1
                        solution.add_new_route()
                        node_index = 1

        customer_to_add = "D0"
        potential_distance_travelled = self.instance.distance_matrix._get_item(
            solution.routes[route_nmb][-1]["name"], 
            customer_to_add
        )
        new_soc = (solution.vehicles[route_nmb]["SoC"]*self.instance.battery_tank_capacity - self.instance.vehicle_consumption_rate * potential_distance_travelled) / self.instance.battery_tank_capacity
        # if needed recharge to arrive to the depot
        if new_soc < 0.1:
            station_to_add = self.get_nearest_station(solution.routes[route_nmb][-1]["name"])
            solution.add_node(station_to_add, route_nmb, node_index)
            node_index += 1
            potential_distance_travelled = self.instance.distance_matrix._get_item(
                solution.routes[route_nmb][-1]["name"],
                customer_to_add
            )
            solution.add_node(customer_to_add, route_nmb, node_index)
        else:
            solution.add_node(customer_to_add, route_nmb, node_index)

        return solution

    def get_nearest_customer(self, start_idx, electric_vehicle, customer_name_lst):
        # find the next customer with the greedy insertion
        infeasible = True
        infeasible_list = []
        # now find the index of the min, excluding the values with 0, needs to be a C
        # check for availability on adding, if the next customer cannot be served because the EV would arrive late,
        # then the customer is ignored. If the next customer would bring the EV over cargo limit, it is ignored.
        # If the stopping conditions are met, the function returns the string "end_time" and the EV prepares for return to depot
        while infeasible: 
            distance_from_start_idx = self.instance.distance_matrix.distance_matrix_df[start_idx]
            # now find the index of the min, excluding the values with 0, needs to be a C
            # remove useless columns from series
            remove_cols = []
            for idx in distance_from_start_idx.axes[0]:
                is_depot = "D" in idx
                is_station = "S" in idx
                if is_depot or is_station or idx in infeasible_list or idx not in customer_name_lst:
                    
                    if idx not in remove_cols:
                        remove_cols.append(idx)
                    
            if distance_from_start_idx.drop(remove_cols).empty:
                return "D0"
            
            # get nearest customer to starting point for the greedy insertion after removing infeasible customers and depot/station
            potential_customer = distance_from_start_idx.drop(remove_cols).idxmin() 
            distance_travelled = self.instance.distance_matrix._get_item(
                start_idx,
                potential_customer
            )
            
            # check for customer feasibility in terms of time, cargo
            for customer in self.instance.customers:
                if potential_customer in customer["StringID"]:
                    due_date = customer["DueDate"]
                    cargo_weigth = customer["demand"]
            if (distance_travelled/self.instance.average_velocity + electric_vehicle["time"]) > due_date or (electric_vehicle["current_cargo"]+cargo_weigth)>self.instance.vehicle_cargo_capacity:
                infeasible_list.append(potential_customer)
            else:
                infeasible = False

        return potential_customer

    def get_nearest_station(self, start_idx):
        depot_point_series = self.instance.distance_matrix.distance_matrix_df[start_idx] # now find the index of the min, excluding the values with 0, needs to be a C
        
        # remove useless columns from series
        remove_cols = []
        for idx in depot_point_series.axes[0]:
            if "D" in idx or "C" in idx:
                try:
                    remove_cols.append(idx)
                except:
                    pass

        if depot_point_series.drop(remove_cols).empty:
            return "end_time" 
        
        potential_station = depot_point_series.drop(remove_cols).idxmin() 
        return potential_station
