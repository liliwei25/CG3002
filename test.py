from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from Support import ConfusionMatrix
from Validation import KFold
import numpy
import pandas
import pickle
from sklearn.externals import joblib


data = numpy.load('./training_data.npy')
#print(td)
labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

data = numpy.delete(data, -1, axis=1)
print(data.shape)

#optimise
#rf
accuracy = []
num_trees = []
best = 0
for i in range (1,51):
    rf_model = RandomForestClassifier(n_estimators=i)
    acc = KFold.get_k_fold_accuracy(rf_model, 10, data, labels)
    if acc > best:
        best = acc
        best_model = rf_model
    num_trees.append(i)
    accuracy.append(acc)
#pickle.dump(best_model, open('./out/rf_model.sav', 'wb'))
joblib.dump(best_model, "rf_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_trees' : num_trees
})
df.to_csv('rf_trees_vs_acc.csv')

# #knn
# accuracy = []
# num_neighbours = []
# best = 0
# for i in range (1,31):
#     knn_model = KNeighborsClassifier(n_neighbors=i)
#     acc = KFold.get_k_fold_accuracy(knn_model, 10, data, labels)
#     if acc > best:
#         best = acc
#         best_model = knn_model
#     num_neighbours.append(i)
#     accuracy.append(acc)
# #pickle.dump(best_model, open('./out_knn/knn_model.sav', 'wb'))
# joblib.dump(best_model, "knn_model.sav")
# df = pandas.DataFrame.from_dict({
#     'accuracy' : accuracy,
#     'num_neighbours' : num_neighbours
# })
# df.to_csv('neighbours_vs_acc.csv')