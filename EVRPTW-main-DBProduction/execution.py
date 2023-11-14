from msilib.schema import Error
import os, csv
from time import time
import csv
import os


# C:/Users/enrin/AppData/Local/Programs/Python/Python39/python.exe "c:/Users/enrin/Desktop/LNS10Seed/EVRPTW-main-DBProduction/execution.py" > output.txt 2>&1

db_Output = open('DB-Output12000_S2000_S1000_hybrid_worse.csv', 'w', newline='')
writer = csv.writer(db_Output)
writer.writerow(["Instance's Name","Iteration","Seed","Initial Solution","OFIS","Moves","OFFS","OF_Diff","Exe_Time_d-r","Avg_Battery_Status","Avg_SoC","Avg_Num_Charge",
                 "Avg_Vehicle_Capacity","Avg_Customer_Demand","Num_Vehicles","Avg_Service_Time","Avg_Customer_TimeWindow","Var_Customer_TimeWindow",
                 "Avg_Customer_customer_min_dist","Var_Customer_customer_min_dist","Avg_Customer_station_min_dist","Var_Customer_station_min_dist",
                 "Avg_Customer_deposit_dist","Var_Customer_deposit_dist","CounterD_R","CounterD_Rlast"])
db_Output.close()

counterinit=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

fC = open('./Counter.csv', 'w')
writer = csv.writer(fC)
writer.writerow(counterinit)
fC.close()

fCL = open('./Counterlast.csv', 'w')
writer = csv.writer(fCL)
writer.writerow(counterinit)
fCL.close()

# Re-initialize ./etc/settings.json
def initialize_file_settings():
    new_file = """{
        "unit_energy_cost" : 0.4,
        "driver_wage" : 1,
        "fixed_vehicle_acquisition" : 1200,
        "overtime_cost_numerator" : 11,
        "overtime_cost_denominator" : 6,
        "rho_low" : 0.3,
        "rho_high" : 0.7,
        "instance_file_name" : "c106_21_100.txt",
        "service_time_generation_type" : "basic",
        "basic_service_time" : {
            "R" : { "low" : 8,
                    "high" : 12},
            "RC" : { "low" : 8,
                    "high" : 12},
            "C" : { "low" : 70,
                    "high" : 1100}
        }
    }"""

    with open('EVRPTW-main-DBProduction\etc\settings.json', 'w') as file:
        file.write(new_file)

def one_hour_running_code():
    files_list = []
    for file in os.listdir('EVRPTW-main-DBProduction\dataTestSeed'):
        files_list.append(file)
    seeds=[123, 42, 29, 3, 18, 7, 11, 25, 9, 14]
    # Makes each instance running 10 times for 10 iterations
    for i in range(len(files_list)):
        for j in range(10): 
            try:
                os.system("python EVRPTW-main-DBProduction\main.py")
                #change seed every of 10 times
                fileR_main = open('EVRPTW-main-DBProduction\main.py', 'r')
                filedata_main = fileR_main.read()
                fileR_main.close()
                if j+1 < len(seeds):
                            filedata_main = filedata_main.replace(str(seeds[j]), str(seeds[j+1]))

                fileW_main = open('EVRPTW-main-DBProduction\main.py', 'w')
                fileW_main.write(filedata_main)
                fileW_main.close()    
                #end change seed     
            except Exception as e:
                print(e)

                pass
        #break

        #reset seed to 123
        fileR_main = open('EVRPTW-main-DBProduction\main.py', 'r')
        filedata_main = fileR_main.read()
        fileR_main.close()

        filedata_main = filedata_main.replace(str(seeds[-1]), str(seeds[0]))

        fileW_main = open('EVRPTW-main-DBProduction\main.py', 'w')
        fileW_main.write(filedata_main)
        fileW_main.close()    
        #end reset 
        fileR = open('EVRPTW-main-DBProduction\etc\settings.json', 'r')
        filedata = fileR.read()
        fileR.close()
        if i + 1 < len(files_list):
            filedata = filedata.replace(files_list[i], files_list[i+1])

        fileW = open('EVRPTW-main-DBProduction\etc\settings.json', 'w')
        fileW.write(filedata)
        fileW.close()

if __name__ == "__main__":

    initialize_file_settings()
    one_hour_running_code()

    
'''    
    #Changing seed from '123' to '42'
    fileR_main = open('EVRPTW-main-DBProduction\main.py', 'r')
    filedata_main = fileR_main.read()
    fileR_main.close()

    filedata_main = filedata_main.replace('123', '42')

    fileW_main = open('EVRPTW-main-DBProduction\main.py', 'w')
    fileW_main.write(filedata_main)
    fileW_main.close()

    initialize_file_settings()

    one_hour_running_code()
'''
