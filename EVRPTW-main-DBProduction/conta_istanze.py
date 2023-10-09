import csv
import os
unique_strings = {}  # Crea un dizionario vuoto per tenere traccia delle stringhe uniche e della loro cardinalità


# FILE GENERATO (non contenente tutte le istanze)
with open('DB-Output8.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        string = row[0]
        if string in unique_strings:
            unique_strings[string] += 1  # Aggiorna la cardinalità se la stringa è già presente nel dizionario
        else:
            unique_strings[string] = 1  # Aggiunge la stringa al dizionario e imposta la sua cardinalità a 1

for string, count in unique_strings.items():
    print(string, count)  # Stampa la stringa e la sua cardinalità


keys_list = list(unique_strings.keys())  # Ottiene la lista di tutte le chiavi del dizionario

done_files=keys_list[1:]

# CARTELLA COMPLETA che contiene tutte le istanze
full_files = []
for file in os.listdir('EVRPTW-main-DBProduction\data'):#oppure data_without_r
    full_files.append(file)

files_mancanti = []
for elem in full_files:
    if elem not in done_files:
        files_mancanti.append(elem)

print(files_mancanti[0:-1]) 
