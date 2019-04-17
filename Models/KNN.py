from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from Support.ConfusionMatrix import print_confusion_matrix
from Validation.KFold import get_k_fold_accuracy

def knn(data, labels):
    print("\nBegin KNN")
    training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size=0.1)
    model = KNeighborsClassifier()  # using default

    print("5-Fold accuracy:", get_k_fold_accuracy(model, 10, data, labels))

    model.fit(training_data, training_labels)
    predicted_labels = model.predict(test_data)

    print_confusion_matrix(test_labels, predicted_labels)

    return model
