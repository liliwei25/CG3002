from Validation.KFold import get_k_fold_accuracy
from Validation.LOOCV import get_loocv_accuracy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Support.ConfusionMatrix import print_confusion_matrix

'''
Inputs:
    kernel: selected kernel (linear, poly, rbf, sigmoid)
    data: array of training and testing data
    labels: array of training and testing labels
Returns:
    returns the trained SVM model

Also prints the 10-fold accuracy and the confusion matrix of the model
'''
def svm(kernel, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    print(f"\nTraining {kernel} SVM classifier... ")
    svc_classifier = SVC(kernel=kernel, gamma=0.001, degree=8)

    # get 5-fold accuracy
    print("5-fold accuracy:", get_k_fold_accuracy(svc_classifier, 5, data, labels))

    # get LOOCV accuracy
    # print("LOOCV accuracy:", get_loocv_accuracy(svc_classifier, data, labels))

    # Train model with 80% data
    svc_classifier.fit(x_train, y_train)
    print(f"\nTesting {kernel} SVM classifier... ")

    # get confusion matrix
    y_pred = svc_classifier.predict(x_test)
    print_confusion_matrix(y_test, y_pred)

    return svc_classifier
