# Import relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np  # For array manipulation
from sklearn.model_selection import GridSearchCV, ShuffleSplit  # For grid search and cross-validation
from sklearn.metrics import accuracy_score  # For evaluating model accuracy
from concrete.ml.sklearn import ConcreteXGBClassifier, SklearnXGBClassifier  # Import your custom classifiers

# %matplotlib inline

# Load in Corporate Credit Rating dataset as a Pandas dataframe

df = pd.read_csv('ratings.csv')

# Visualise dataset

df.head()

# 'rating' is the response variable


param_grid = {
  
  "max_depth": list(range(1, 5)),
  
  "n_estimators": list(range(1, 201, 20)),  
  
  "learning_rate": [0.01, 0.1, 1],
  
  "n_bits": [4]  # 'n_bits' controls how many bits are used to quantise each value; generally more bits mean better accuracy, but slower to run   
} 

# We use shuffle split for cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# Create a grid search variable and pass in the relevant arguments,
# then fit the variable with the training data

concrete_grid_search = GridSearchCV(
    ConcreteXGBClassifier(), param_grid, cv=cv, scoring='roc_auc'
)
concrete_grid_search.fit(X_train, y_train)

# Set of optimal parameters for 
concrete_best_params = concrete_grid_search.best_params_

# Now define the actual XGBoost model using optimised parameters
concrete_model = ConcreteXGBClassifier(**concrete_best_params)


# Training the actual model with the training data
concrete_model.fit(X_train, y_train)

# Compile the model to generate a FHE circuit
concrete_model.compile(X_train[100:])

# Generate an array containing the random permutation of integers in [0,49]
n_sample_to_test_fhe = 50
idx_test = np.random.choice(X_test.shape[0], n_sample_to_test_fhe, replace=False)

# This is so we can select a small random sample of size 'n_sample_to_test_fhe'
# in FHE, for a relatively quick test of model accuracy
X_test_fhe = X_test[idx_test]
y_test_fhe = y_test[idx_test]

# Train the same model from sklearn, and evaluate the accuracy
# The below procedure is the same as was done for the Concrete ML model

param_grid_sklearn = {
  
  "max_depth": list(range(1, 5)),
  
  "n_estimators": list(range(1, 201, 20)),  # Tune the number of decision trees used in XGBoost. Default is 100
  
  "learning_rate": [0.01, 0.1, 1],
  
  "eval_metric": ["logloss"]
} 

sklearn_grid_search = GridSearchCV(
    SklearnXGBClassifier(), param_grid_sklearn, cv=cv, scoring='roc_auc'
).fit(X_train, y_train)

sklearn_best_params = sklearn_grid_search.best_params_
sklearn_model = SklearnXGBClassifier(**sklearn_best_params)
sklearn_model.fit(X_train, y_train)

y_pred_clear = sklearn_model.predict(X_test_fhe)  # Regular plaintext prediction

y_pred_clear_q = concrete_model.predict(X_test_fhe)  # Quantised plaintext prediction

y_preds_fhe = concrete_model.predict(X_test_fhe, execute_in_fhe=True)  # FHE prediction

# Use Sklearn's built in accuracy evaluation function to compare accuracies of the different modes

from sklearn.metrics import accuracy_score

print(f'Accuracy score of clear model: {accuracy_score(y_test_fhe, y_pred_clear)}')
print(f'Accuracy score of clear quantised model: {accuracy_score(y_test_fhe, y_pred_clear_q)}')
print(f'Accuracy score of FHE model: {accuracy_score(y_test_fhe, y_preds_fhe)}')
