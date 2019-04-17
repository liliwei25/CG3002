import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from Support.Data import import_data
from Support.DataPreprocessing import preprocess

data = import_data()
training_data = []
x = data[:, :-1]
y = data[:, -1]
x = preprocess(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)


c_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=c_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(x, y)

print(f"Best parameters are {grid.best_params_} with a score of {grid.best_score_}")
