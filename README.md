# Model Evaluation and Validation
## Project: Predicting Boston Housing Prices
### Files Description
- `project description.md`: Project overview, highlights, evaluation and software requirement. 
- `README.md`: this file.
- `boston_housing.ipynb`: This is the main file that will be performing my work on the project.
- `housing.csv`: The project dataset. I'll load this data in the notebook.
- `visuals.py`: This Python script provides supplementary visualizations for the project. **Not my work**
- `boston_housing.html`: `html` version of the main file.

### Run
#### 1.Want Modify
In a command window (OS: Win7), navigate to the top-level project directory that contains this README and run one of the following commands:
`jupyter notebook boston_housing.ipynb`
This will open the Jupyter Notebook software and project file in your browser.
#### 2.Just Have a Look
Double click `boston_housing.html` file. You can see this file in your browser.

## Project Implementation
### Data Exploration
#### Implementation: Calculate Statistics
Use `numpy` to calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
Store each calculation in their respective variable.
```
# TODO: Minimum price of the data
minimum_price = np.amin(prices)
#minimum_price = prices.min()

# TODO: Maximum price of the data
maximum_price = np.amax(prices) #prices.max()

# TODO: Mean price of the data
mean_price = np.mean(prices) #prices.mean()

# TODO: Median price of the data
median_price = np.median(prices) #prices.median()

# TODO: Standard deviation of prices of the data
std_price = np.std(prices) #prices.std()
```


#### Feature Observation
Correctly justify how each feature correlates with an increase or decrease in the target variable.
### Developing a Model
#### Implementation: Define a Performance Metric
Calculate the [coefficient of determination](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), $R^2$, to quantify  model's performance.
Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
Assign the performance score to the `score` variable.
```
# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict) 
    
    # Return the score
    return score
```


#### Goodness of Fit
Correctly identify whether the hypothetical model successfully captures the variation of the target variable based on the model’s R^2 score.
#### Implementation: Shuffle and Split Data
Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
Split the data into 80% training and 20% testing.
Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.
```
# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)
```
#### Training and Testing
Provide a valid reason for why a dataset is split into training and testing subsets for a model. 
### Analyzing Model Performance
#### Learning the Data
- Correctly identify the trend of both the training and testing curves from the graph as more training points are added. 
- Discussion is made as to whether additional training points would benefit the model.

#### Bias-Variance Tradeoff
Correctly identify whether the model at a max depth of 1 and a max depth of 10 suffer from either high bias or high variance, with justification using the complexity curves graph.
#### Best-Guess Optimal Model
Pick a best-guess optimal model with reasonable justification using the model complexity graph.
### Evaluating Model Performance
#### Grid Search
Correctly describe the grid search technique and how it can be applied to a learning algorithm.
#### Cross-Validation
Correctly describe the k-fold cross-validation technique and discuss the benefits of its application when used with grid search when optimizing a model.
#### Implementation: Fitting a Model
Use the grid search technique to optimize the `'max_depth'` parameter for the **decision tree algorithm**
- Use `DecisionTreeRegressor` from `sklearn.tree` to create a decision tree regressor object.
    Assign this object to the `'regressor'` variable.
- Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
- Use `make_scorer` from `sklearn.metrics` to create a scoring function object.
    Pass the `performance_metric` function as a parameter to the object.
    Assign this scoring function to the 'scoring_fnc' variable.
- Use `GridSearchCV` from `sklearn.grid_search` to create a grid search object.
    Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object.
    Assign the `GridSearchCV` object to the `'grid'` variable.
```
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    #cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
```
#### Predicting Selling Prices
Predict selling price for the three clients listed in the provided table. Discussion is made for each of the three predictions as to whether these prices are reasonable given the data and the earlier calculated descriptive statistics.
#### Applicability
Discuss whether the model should or should not be used in a real-world setting.