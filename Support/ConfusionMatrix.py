import numpy as np
from sklearn.metrics import confusion_matrix

'''
Inputs:
    y_test = expected labels
    y_pred = predicted labels
Output:
    No return output.
    Prints non-normalized and normalized confusion matrix.
    Prints accuracy calculated from confusion matrix.
'''
def print_confusion_matrix(y_test, y_pred, labels=None):
    con_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (non-normalized): ")
    print(con_matrix)
    print("\nConfusion Matrix (normalized): ")
    print(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis])
    print("\nCalculated accuracy:", np.sum(np.diagonal(con_matrix))/ np.sum(con_matrix))