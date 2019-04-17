from Models.KNN import knn
from Models.RandomForest import random_forest, random_forest_opt
from Support.Data import import_data
from Models.SVM import svm
from Models.NaiveBayes import naive_bayes
from Models.ANN import neural_network
from sklearn.externals import joblib
from sklearn.utils import shuffle
import sklearn
import numpy as np
import csv

from Support.DataPreprocessing import preprocess_data, preprocess, process_files

svm_kernels = {1: "linear", 2: "poly", 3: "rbf", 4: "sigmoid"}
nb_classifiers = {1: "gaussian", 2: "multinomial", 3: "complement", 4: "bernoulli"}


if __name__ == '__main__':
    # import data
    # print(sklearn.__version__)
    data = import_data()
    print(data.shape)
    data = process_files(data)
    # data = np.load('./training_data.npy')
    np.save("training_data.npy", data)
    x = data[:, :-1]
    y = data[:, -1]
    print(y)
    # y = data[:, -1]
    # x = preprocess(x)


    # ARSCMA raw
    # for i in range(1, 8):
    #     x = data[:-1,data[3]==i]
    #     x = preprocess_data(x, i)
    #     training_data.extend(x)
    # with open("wisdm_reduced.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(training_data)
    # exit(0)


    # WISDM raw
    # for i in range(0, 6):
    #     x = data[1:,data[0]==i]
    #     x = preprocess_data(x, i)
    #     training_data.extend(x)
    # with open("wisdm_reduced.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(training_data)
    # exit(0)

    # data = data[data[:,-1] !=0]
    # data = shuffle(data)
    # x = data[:, :-1]
    # y = data[:, -1]

    print("Training data:", x.shape)
    print("Data Labels:", y.shape)


    model = input("Choose model (svm, nb, rf, knn, ann): ")
    if model == "svm":
        print("SVM is chosen... ")
        kernel = input("Choose Kernel (1: linear, 2: poly, 3: gaussian/rbf, 4: sigmoid): ")
        trained_model = svm(svm_kernels[int(kernel)], x, y)
    elif model == "nb":
        print("Naive Bayes is chosen...")
        classifier = input("Choose classifier (1: gaussian, 2: multinomial, 3: complement, 4: bernoulli): ")
        trained_model = naive_bayes(nb_classifiers[int(classifier)], x, y)
    elif model == "rf":
        print("Random Forest is chosen...")
        trained_model = random_forest(x, y)
    elif model == "knn":
        print("KNN is chosen...")
        trained_model = knn(x, y)
    else:
        print("ANN is chosen...")
        trained_model = neural_network(x, y)
    joblib.dump(trained_model, f"trained_model_{model}_all.sav")
