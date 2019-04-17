from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

'''
Inputs:
    model: Machine Learning Model
    data: Training and testing data
    labels: Labels for data
Output:
    Average accuracy from LOOCV
'''
def get_loocv_accuracy(model, data, labels):
    loo = LeaveOneOut()
    print(f"\nCalculating LOOCV accuracy with {loo.get_n_splits(data)} iterations...")
    total_accuracy = 0
    for training_indices, testing_indices in loo.split(data):
        model.fit(data[training_indices], labels[training_indices])
        y_predicted = model.predict(data[testing_indices])
        total_accuracy += accuracy_score(labels[testing_indices], y_predicted)
    return total_accuracy / loo.get_n_splits(data)
