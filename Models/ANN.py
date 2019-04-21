from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from Validation.KFold import get_k_fold_accuracy
from Support.ConfusionMatrix import print_confusion_matrix

def neural_network(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    print(f"\nTraining ANN classifier... ")
    ann_classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(400,))

    # get 5-fold accuracy
    print("5-fold accuracy:", get_k_fold_accuracy(ann_classifier, 5, data, labels))

    # get LOOCV accuracy
    # print("LOOCV accuracy:", get_loocv_accuracy(svc_classifier, data, labels))

    # Train model with 80% data
    ann_classifier.fit(x_train, y_train)
    print(f"\nTesting ANN classifier... ")

    # get confusion matrix
    y_pred = ann_classifier.predict(x_test)
    print_confusion_matrix(y_test, y_pred)

    return ann_classifier