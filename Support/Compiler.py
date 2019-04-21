import os
from Support.Data import Move, dump_first_half
import numpy as np

def get_every_3(data):
    new_data = []
    for row, row_data in enumerate(data):
        if row % 3 == 0:
            new_data.append(row_data)
    return new_data

def preprocess_data(file):
    temp = file[0]
    for data in file[1:]:
        temp = np.append(temp, dump_first_half(data), axis=0)
    return temp

for file in os.listdir("../Data/Training Data/No delay"):
    if file.endswith(".npy"):
        for mv in Move:
            if mv.name in file:
                data_file = np.load("../Data/Training Data/No delay/" + file)
                data_array = preprocess_data(data_file)
                data_array = get_every_3(data_array)
                np.save("../Data/Training Data/No delay/3/" + file.replace(mv.name, mv.name+"_3"), data_array)
                print(f"Saved to {file.replace(mv.name, mv.name+'_3')}")









