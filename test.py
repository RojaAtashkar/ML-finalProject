
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
# import the data required
df = pd.read_csv('train.csv')
df = pd.get_dummies(df)
y = df[df.columns[-1]]
print(df.columns[-1])
X = df.drop(df.columns[-1], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)
print(X_train)
mlp = MLPClassifier(max_iter=10000)
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(10, 20, 20), (20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant','adaptive'],
}



grid = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
mlp_param = grid.best_params_
mlp_estimator = grid.best_estimator_
mlp_estimator.fit(X_train, y_train)
y_pred_mlp = mlp_estimator.predict(X_test)
print(classification_report(y_test, y_pred_mlp))