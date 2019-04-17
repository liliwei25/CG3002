from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

'''
Inputs:
    model: Machine Learning Model
    folds: Number of folds (K)
    data: Training and testing data
    labels: Labels for data
Output:
    Average accuracy from the K Folds
'''
def get_k_fold_accuracy(model, folds, data, labels):
    print(f"\nCalculating KFold accuracy with {folds} iterations...")
    kf = KFold(folds, shuffle=True)
    total_accuracy = 0
    for training_indices, testing_indices in kf.split(data):
        model.fit(data[training_indices], labels[training_indices])
        y_predicted = model.predict(data[testing_indices])
        total_accuracy += accuracy_score(labels[testing_indices], y_predicted)
    return total_accuracy/folds