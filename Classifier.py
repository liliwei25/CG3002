from Support.DataPreprocessing import get_features_from_frame
from time import  time
import pandas as pd
import numpy as np
from sklearn.externals import joblib

# trained_model = load("trained_model_svm.sav")
trained_model = joblib.load("trained_model_rf_all.sav")


def predict(data):
    start = time()
    data = get_features_from_frame(data)
    data = np.array(data).reshape(1, -1)
    print(trained_model.predict(data)[0])
    print(trained_model.predict_proba(data)[0])
    print(f"Time taken: {time() - start}s")


# gyro_x = pd.read_csv("./Data/HAR/body_gyro_x.csv", engine="python").values
# gyro_y = pd.read_csv("./Data/HAR/body_gyro_y.csv", engine="python").values
# gyro_z = pd.read_csv("./Data/HAR/body_gyro_z.csv", engine="python").values
# acc_x = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
# acc_y = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
# acc_z = pd.read_csv("./Data/HAR/total_acc_x.csv", engine="python").values
# y = pd.read_csv("./Data/HAR/y.csv", engine="python").values
# data = np.concatenate([gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, y], axis=1)





data = np.load("./Data/Training Data/22-3-19-25856_RAFFLESL20SI0.05R0.5(0).npy")

# Should output 2
predict(data[0])