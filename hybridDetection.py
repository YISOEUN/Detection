import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------------------------------------------------------------------------------------
# misuse detection
misuse_file = 'C:/Users/samsung/Desktop/Dataset_Misuse_AttributeSelection.csv'
misuse_df = pd.read_csv(misuse_file)
print(misuse_df)

# fill nan
misuse_df = misuse_df.replace({'AttackType':"NA"},{'AttackType':"Normal"})
misuse_df = misuse_df.fillna('Normal')

misuse_x = misuse_df.drop('AttackType', axis=1)
misuse_y = misuse_df['AttackType'].values

# anomaly detection
anomaly_file = 'C:/Users/samsung/Desktop/testData.csv'
anomaly_df = pd.read_csv(anomaly_file)
print(anomaly_df)

# Data encoding
encoder = LabelEncoder()

# encoding
anomaly_df['protocol_type'] = encoder.fit_transform(anomaly_df['protocol_type'])
anomaly_df['service'] = encoder.fit_transform(anomaly_df['service'])
anomaly_df['flag'] = encoder.fit_transform(anomaly_df['flag'])

anomaly_x = anomaly_df.drop('class', axis=1)
anomaly_y = anomaly_df['class'].values

# Read test data
test_file = 'C:/Users/samsung/Desktop/Dataset_Anomaly_AttributeSelection.csv'
test_df = pd.read_csv(test_file)
print(test_df)
test_df = test_df.drop('AttackType', axis=1)

# Scale the data
# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
test_X = MinMax_scaler.fit_transform(test_df)

# -------------------------------------------------------------------------------------------------------------
logistic_params = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs', 'sag'],
    'max_iter': [50, 100, 200]
}
decisionTree_params = {
    'max_depth': [None, 6, 8, 10, 12, 16, 20, 24],
    'min_samples_split': [2, 20, 50, 100, 200],
    'criterion': ['entropy', 'gini']
}
randomForest_params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

gradient_params = {
    "n_estimators": [50,100],
    "max_depth": [1, 2, 4],
    "learning_rate": [0.01, 0.1]
}

# -------------------------------------------------------------------------------------------------------------
# Scale the data
print("Misuse detection")
X_scaled = MinMax_scaler.fit_transform(misuse_x)

# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, misuse_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, misuse_y)
randomForest = RandomForestClassifier().fit(X_scaled, misuse_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, misuse_y)

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)  # print best parameter value
print('best score', gcv_logistic.best_score_)  # best score
logistic_best = gcv_logistic.best_estimator_
gcv_logistic_score = gcv_logistic.best_score_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)  # print best parameter value
print('best score', gcv_decisionTree.best_score_)  # best score
gcv_decisionTree_score = gcv_decisionTree.best_score_
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)  # print best parameter value
print('best score', gcv_randomForest.best_score_)  # best score
gcv_randomForest_score = gcv_randomForest.best_score_
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
print('best score', gcv_gradientBoosting.best_score_)  # best score
gradientBoosting_best = gcv_gradientBoosting.best_estimator_
gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_
gradientBoosting_predict = gradientBoosting_best.predict(test_X)
print(gradientBoosting_predict)

# Find best model & score
score = [gcv_decisionTree_score, gcv_randomForest_score, gcv_gradientBoosting_score]
models = [decisionTree_best, randomForest_best,gradientBoosting_best]
max_score = gcv_logistic_score
best_model = logistic_best
for i in range(len(score)):
    if max_score < score[i]:
        max_score = score[i]
        best_model = models[i]

print("Misuse detection best score: ",max_score)

# a new dataframe by collecting only normal results through best model
normal_X = pd.DataFrame(columns=test_df.columns)

index = 0
result = list(best_model.predict(test_X))
for i in range(len(result)):
    if result[i] == 'Normal':
        normal_X.loc[index] = test_X[i]
        index = index + 1

print(normal_X)

# -------------------------------------------------------------------------
print("Anomaly detection")
value = anomaly_x.columns
X_scaled = MinMax_scaler.fit_transform(anomaly_x)

# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, anomaly_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, anomaly_y)
randomForest = RandomForestClassifier().fit(X_scaled, anomaly_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, anomaly_y)

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)  # print best parameter value
print('best score', gcv_logistic.best_score_)  # best score
logistic_best = gcv_logistic.best_estimator_
gcv_logistic_score = gcv_logistic.best_score_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)  # print best parameter value
print('best score', gcv_decisionTree.best_score_)  # best score
decisionTree_best = gcv_decisionTree.best_estimator_
gcv_decisionTree_score = gcv_decisionTree.best_score_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)  # print best parameter value
print('best score', gcv_randomForest.best_score_)  # best score
randomForest_best = gcv_randomForest.best_estimator_
gcv_randomForest_score = gcv_randomForest.best_score_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
print('best score', gcv_gradientBoosting.best_score_)  # best score
gradientBoosting_best = gcv_gradientBoosting.best_estimator_
gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_

# best model & score
score = [gcv_decisionTree_score, gcv_randomForest_score,gcv_gradientBoosting_score]
models = [decisionTree_best, randomForest_best,gradientBoosting_best]
max_score = gcv_logistic_score
best_model = logistic_best
for i in range(len(score)):
    if max_score < score[i]:
        max_score = score[i]
        best_model = models[i]

print("Anomaly detection best score: ",max_score)

result = list(best_model.predict(normal_X))
for i in range(len(result)) :
    if result[i] == 'anomaly' :
        print('index',i,'---->',result[i])
        print(normal_X.iloc[i])
