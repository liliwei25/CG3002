from sklearn.model_selection import train_test_split

from Support.ConfusionMatrix import print_confusion_matrix
from Validation.KFold import get_k_fold_accuracy
from Validation.LOOCV import get_loocv_accuracy

def naive_bayes(classifier, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    print(f"\nTraining {classifier} NB classifier... ")
    if classifier == "gaussian":
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
    elif classifier == "multinomial":
        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB()
    elif classifier == "complement":
        from sklearn.naive_bayes import ComplementNB
        nb = ComplementNB()
    elif classifier == "bernoulli":
        from sklearn.naive_bayes import BernoulliNB
        nb = BernoulliNB()
    else:
        return

    # 5-fold
    print("5-fold accuracy:", get_k_fold_accuracy(nb, 10, data, labels))

    # LOOCV
    # print("LOOCV accuracy:", get_loocv_accuracy(nb,data,labels))

    # Train model with 80% data
    nb.fit(x_train, y_train)
    print(f"\nTesting {classifier} NB classifier... ")

    # get confusion matrix
    y_pred = nb.predict(x_test)
    print_confusion_matrix(y_test, y_pred)

    return nb
