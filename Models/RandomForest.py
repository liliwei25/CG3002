from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from Support.ConfusionMatrix import print_confusion_matrix
from Validation.KFold import get_k_fold_accuracy


def random_forest(data, labels):
    print("\nBegin Random Forest Classifier")
    training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size=0.1)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # using default

    print("5-Fold accuracy:", get_k_fold_accuracy(model, 5, data, labels))

    # model.fit(data, labels)
    model.fit(training_data, training_labels)
    predicted_labels = model.predict(test_data)

    print_confusion_matrix(test_labels, predicted_labels)
    # LOOCV.get_loocv_accuracy(model, self.data, self.labels)

    return model

def random_forest_opt(data, labels):
    print("\nBegin Random Forest Classifier")
    training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size=0.1)

    best_acc = 0
    best_model = None
    for i in range(5, 31):
        model = RandomForestClassifier(n_estimators=i)  # using default
        acc = get_k_fold_accuracy(model, 5, data, labels)
        if acc >= best_acc:
            best_model = model
            best_acc = acc
    print("acc for best num of trees: ", best_acc)
    params = best_model.get_params()
    print(params)
    print(str(params['n_estimators']))

    best_model.fit(training_data, training_labels)
    predicted_labels = best_model.predict(test_data)

    print_confusion_matrix(test_labels, predicted_labels)
    return best_model

