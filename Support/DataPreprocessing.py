from Support.Sampler import Sampler
from scipy.stats import iqr, entropy
import pandas as pd
import numpy as np


def preprocess_data(data, index):
    sampler = Sampler(3, 128, 64)
    frames = sampler.sample(data)
    data = []
    for frame in frames:
        x = []
        for row in frame:
            x.extend(np.append(row, [np.mean(row), np.var(row), np.median(row), pd.np.std(row)]))
        x.append(index)
        data.append(x)
    return data

def process_files(files):
    data = []
    label = []
    for file in files:
        features = process_frames(file[0])
        data.extend(features)
        label.extend(np.full((len(features), 1), file[1]))
    data = np.array(data)
    data = np.concatenate([data, label], axis=1)
    return data


def process_frames(frames):
    data = []
    for frame in frames:
        features = get_features_from_frame(frame)
        data.append(np.array(features))
    data = np.array(data)
    return data

def get_features_from_frame(frame):
    data = []
    for i in range(0, frame.shape[1]):
        data = np.append(data, get_features(frame[:, i]))
    return data

def get_features(value):
    return [
        np.mean(value),             # mean
        np.var(value),              # var
        np.median(value),           # median
        iqr(value),                 # iqr
        np.std(value),              # std
        np.max(value),              # max
        np.min(value),              # min
        mad(value)                  # mad
    ]


# def get_features(row):
#     gyro_x = row[0:128]
#     gyro_y = row[128:256]
#     gyro_z = row[256:384]
#     acc_x = row[384:512]
#     acc_y = row[512:640]
#     acc_z = row[640:768]
#
#     return [
#             # len(np.where(np.diff(np.sign(gyro_x)))[0]), len(np.where(np.diff(np.sign(gyro_y)))[0]),
#             # len(np.where(np.diff(np.sign(gyro_z)))[0]),                 # zero crossing of gyro x, y, z
#             # len(np.where(np.diff(np.sign(acc_x)))[0]), len(np.where(np.diff(np.sign(acc_y)))[0]),
#             # len(np.where(np.diff(np.sign(acc_z)))[0])#,                  # zero crossing of acc x, y, z
#             np.mean(gyro_x), np.mean(gyro_y), np.mean(gyro_z),          # mean of gyro x, y, z
#             np.mean(acc_x), np.mean(acc_y), np.mean(acc_z),             # mean of acc x, y, z
#             np.var(gyro_x), np.var(gyro_y), np.var(gyro_z),             # var of gyro x, y, z
#             np.var(acc_x), np.var(acc_y), np.var(acc_z),                # var of acc x, y, z
#             np.median(gyro_x), np.median(gyro_y), np.median(gyro_z),    # median of gyro x, y, z
#             np.median(acc_x), np.median(acc_y), np.median(acc_z),       # median of acc x, y, z
#             iqr(gyro_x), iqr(gyro_y), iqr(gyro_z),                      # iqr of gyro x, y, z
#             iqr(acc_x), iqr(acc_y), iqr(acc_z),                         # iqr of acc x, y, z
#             np.std(gyro_x), np.std(gyro_y), np.std(gyro_z),             # std of gyro x, y, z
#             np.std(acc_x), np.std(acc_y), np.std(acc_z),                # std of acc x, y, z
#             np.max(gyro_x), np.max(gyro_y), np.max(gyro_z),             # max of gyro x, y, z
#             np.max(acc_x), np.max(acc_y), np.max(acc_z),                # max of acc x, y, z
#             np.min(gyro_x), np.min(gyro_y), np.min(gyro_z),             # min of gyro x, y, z
#             np.min(acc_x), np.min(acc_y), np.min(acc_z),                # min of acc x, y, z
#             mad(gyro_x), mad(gyro_y), mad(gyro_z),                      # mad of gyro x, y, z
#             mad(acc_x), mad(acc_y), mad(acc_z),                         # mad of acc x, y, z
#             # np.correlate(gyro_x, gyro_x), np.correlate(gyro_y, gyro_y),
#             # np.correlate(gyro_z, gyro_z),                               # auto-correlation of gyro x, y, z
#             # np.correlate(acc_x, acc_x), np.correlate(acc_y, acc_y),
#             # np.correlate(acc_z, acc_z),                                 # auto-correlation of acc x, y, z
#             # np.correlate(gyro_x, gyro_y), np.correlate(gyro_x, gyro_z), np.correlate(gyro_y, gyro_z),
#             # np.correlate(acc_x, acc_y), np.correlate(acc_x, acc_z), np.correlate(acc_y, acc_z)
#                                         ]

def preprocess(data):
    training = []
    for row in data:
        # fft = (np.fft.fft(row)**2).real

        training.append(np.append([],get_features(row)))

    # x = preprocessing.normalize(x)
    # x, y = shuffle(x, y)
    # print(x.shape)
    # print(np.mean(x).shape)
    # x = np.concatenate([x, np.mean(x, axis=0)], axis=0)
    # print(x)
    return np.array(training)

def mad(data):
    return np.mean(np.absolute(data - np.mean(data)))