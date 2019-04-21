import pandas as pd
import numpy as np
import os
from enum import Enum

class Move(Enum):
    FINAL = 0
    HUNCHBACK = 1
    RAFFLES = 2
    CHICKEN = 3
    CRAB = 4
    COWBOY = 5
    RUNNINGMAN = 6
    JAMESBOND = 7
    SNAKE = 8
    DOUBLEPUMP = 9
    MERMAID = 10

def import_data():
    print("Importing data... \n")

    # HAR dataset
    # Features
    # x_train = pd.read_csv("./Data/HAR/x_train.csv", engine="python").values
    # y_train = pd.read_csv("./Data/HAR/y_train.csv", engine="python").values
    # x_test = pd.read_csv("./Data/HAR/x_test.csv", engine="python").values
    # y_test = pd.read_csv("./Data/HAR/y_test.csv", engine="python").values
    # x = np.concatenate([x_train, x_test])
    # y = np.concatenate([y_train, y_test])
    # data = np.concatenate([x, y], axis=1)

    # Raw
    # gyro_x = pd.read_csv("./Data/HAR/body_gyro_x.csv", engine="python").values
    # gyro_y = pd.read_csv("./Data/HAR/body_gyro_y.csv", engine="python").values
    # gyro_z = pd.read_csv("./Data/HAR/body_gyro_z.csv", engine="python").values
    # acc_x = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
    # acc_y = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
    # acc_z = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
    # y = pd.read_csv("./Data/HAR/y.csv", engine="python").values
    # data = np.concatenate([gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, y], axis=1)


    # ARSCMA dataset
    # raw
    # data = pd.read_csv("./Data/ARSCMA/1.csv", engine="python").values
    # for i in range(2, 16):
    #     new_data = pd.read_csv(f"./Data/ARSCMA/{i}.csv", engine="python").values
    #     data = np.concatenate([data, new_data])
    # data = np.transpose(data[:, 1:].astype(int))
    # data = data[:,data[3,:].argsort()]

    # features
    # data = pd.read_csv("./arscma.csv", engine="python").values

    # WISDM dataset
    # raw
    # data = pd.read_csv("./Data/WISDM/1.csv", engine="python").values
    # data = np.concatenate([data, pd.read_csv("./Data/WISDM/2.csv", engine="python").values])
    # data = np.transpose(data)
    # data = data[:, np.invert(np.isnan(data[0]))]
    # data = data[:, data[0, :].argsort()]

    # features
    # data = pd.read_csv("./wisdm.csv", engine="python").values

    # raw (only actions 0, 1, 2)
    # data = pd.read_csv("./Data/WISDM/combined.csv", engine="python").values
    # data = np.transpose(data)
    # data = data[:, np.invert(np.isnan(data[0]))]
    # data = data[:, data[0, :].argsort()]

    # features (only actions 0, 1, 2)
    # data = pd.read_csv("./wisdm_reduced.csv", engine="python").values

    data = []
    for file in os.listdir("./Data/Training Data"):
        if file.endswith(".npy"):
            for mv in  Move:
                if mv.name in file:
                    data_file = np.load("./Data/Training Data/" + file)
                    data.append((preprocess_data(data_file), mv.value))
                    # print(file, mv.name, mv.value)
    for file in os.listdir("./Data/Training Data/No delay"):
        if file.endswith(".npy"):
            for mv in  Move:
                if mv.name in file:
                    data_file = np.load("./Data/Training Data/No delay/" + file)
                    data.append((preprocess_data(data_file), mv.value))
    # for file in os.listdir("./Data/Training Data/No delay/3"):
    #     if file.endswith(".npy"):
    #         for mv in  Move:
    #             if mv.name in file:
    #                 data_file = np.load("./Data/Training Data/No delay/3/" + file)
    #                 data.append((sliding_window(data_file), mv.value))
    # np.save("data.npy", np.array(data))

    # data = np.load('./training_data.npy')

    return np.array(data)


def preprocess_data(file):
    data_array = []
    temp = file[0]
    for data in file[1:]:
        temp = np.append(temp, dump_first_half(data), axis=0)
    data_array.extend(sliding_window(temp))
    return data_array



def dump_first_half(data):
    return data[int(data.shape[0]/2):]

def sliding_window(data):
    windows = np.array([data[:20]])
    for i in range(21, data.shape[0]):
        windows = np.append(windows, [data[i - 20:i]], axis=0)
    return windows

