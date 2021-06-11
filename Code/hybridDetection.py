import warnings

warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import plot_roc_curve

#for pie chart
import matplotlib.pyplot as plt

# for ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# function to draw roc curve
def rocvis(true, prob, label):
    from sklearn.metrics import roc_curve
    if type(true[0]) == str:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true = le.fit_transform(true)
        prob = le.fit_transform(prob)
    else:
        pass
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker='.', label=label)

def hybridDetect(test_file) :
    plt.cla()
    # misuse detection
    misuse_file = './preprocessed_data/Dataset_Misuse_AttributeSelection.csv'
    misuse_df = pd.read_csv(misuse_file)
    print(misuse_df)

    # fill nan
    misuse_df = misuse_df.replace({'AttackType': "NA"}, {'AttackType': "Normal"})
    misuse_df = misuse_df.fillna('Normal')

    misuse_x = misuse_df.drop('AttackType', axis=1)
    misuse_y = misuse_df['AttackType'].values

    # attacks('attack name' : number of each attack)
    attacks = {'Normal':0,'Back':0, 'FTPWrite':0, 'Neptune':0, 'PortSweep':0,'Satan':0, 'BufferOverflow':0,
               'GuessPassword':0, 'NMap':0, 'Rootkit':0, 'Smurf':0}

    misuse_result = ""

    # ------------------------------------------------------
    # anomaly detection
    anomaly_file = './preprocessed_data/Dataset_Anomaly_AttributeSelection.csv'
    anomaly_df = pd.read_csv(anomaly_file)
    anomaly_df = anomaly_df.drop('Unnamed: 0', axis=1)

    anomaly_result = ""

    print(anomaly_df)

    anomaly_x = anomaly_df.drop('AttackType', axis=1)
    anomaly_y = anomaly_df['AttackType'].values

    attack_num = 0
    normal_num = 0

    # ------------------------------------------------------
    # Read test data

    test_df = pd.read_csv(test_file)
    # test_df = test_df.drop('Unnamed: 0', axis=1)
    print(test_df)

    test_y = test_df['AttackType'].values
    test_df = test_df.drop('AttackType', axis=1)

    for i in range(len(test_y)):
        if test_y[i] != 'Normal':
            test_y[i] = 'Attack'

    # Scale the data
    # Scaling - MinMaxScaler, Normalizer, StandardScaler
    MinMax_scaler = MinMaxScaler()
    Standard_scaler = StandardScaler()
    test_X = MinMax_scaler.fit_transform(test_df)

    # -------------------------------------------------------------------------------------------------------------
    # Parameter sets for each Classification models
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
        "n_estimators": [50, 100],
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
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # Logistic Regression
    gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_logistic.fit(X_scaled, misuse_y)
    print("---------------------------------------------------------------")
    print("Logistic Regression")
    print('final params', gcv_logistic.best_params_)  # print best parameter value
    print('best score', gcv_logistic.best_score_)  # best score
    logistic_best = gcv_logistic.best_estimator_
    gcv_logistic_score = gcv_logistic.best_score_

    index = 0

    result_lr = list(logistic_best.predict(test_X))
    for i in range(len(result_lr)):
        if result_lr[i] != 'Normal':
            result_lr[i] = 'Attack'

    # Confusion Matrix & ROC curve
    print(confusion_matrix(test_y, result_lr))
    print(classification_report(test_y, result_lr, target_names=['Attack', 'Normal']))

    prob_lr = logistic_best.predict_proba(test_X)
    rocvis(test_y, prob_lr[:,1], "Logistic")

    # Decision Tree
    gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_decisionTree.fit(X_scaled, misuse_y)
    print("---------------------------------------------------------------")
    print("Decision Tree")
    print('final params', gcv_decisionTree.best_params_)  # print best parameter value
    print('best score', gcv_decisionTree.best_score_)  # best score
    gcv_decisionTree_score = gcv_decisionTree.best_score_
    decisionTree_best = gcv_decisionTree.best_estimator_

    index = 0
    result_dt = list(decisionTree_best.predict(test_X))
    for i in range(len(result_dt)):
        if result_dt[i] != 'Normal':
            result_dt[i] = 'Attack'

    # Confusion Matrix & ROC curve
    print(confusion_matrix(test_y, result_dt))
    print(classification_report(test_y, result_dt, target_names=['Attack', 'Normal']))

    prob_dt = decisionTree_best.predict_proba(test_X)
    rocvis(test_y, prob_dt[:, 1], "DecisionTree")

    # Random Forest
    gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_randomForest.fit(X_scaled, misuse_y)
    print("---------------------------------------------------------------")
    print("Random Forest")
    print('final params', gcv_randomForest.best_params_)  # print best parameter value
    print('best score', gcv_randomForest.best_score_)  # best score
    gcv_randomForest_score = gcv_randomForest.best_score_
    randomForest_best = gcv_randomForest.best_estimator_

    index = 0
    result_rf = list(decisionTree_best.predict(test_X))
    for i in range(len(result_rf)):
        if result_rf[i] != 'Normal':
            result_rf[i] = 'Attack'

    # Confusion Matrix & ROC curve
    print(confusion_matrix(test_y, result_rf))
    print(classification_report(test_y, result_rf, target_names=['Attack', 'Normal']))

    prob_rf = randomForest_best.predict_proba(test_X)
    rocvis(test_y, prob_rf[:, 1], "RandomForest")

    # Gradient Boosting
    gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_gradientBoosting.fit(X_scaled, misuse_y)
    print("---------------------------------------------------------------")
    print("Gradient Boosting")
    print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
    print('best score', gcv_gradientBoosting.best_score_)  # best score
    gradientBoosting_best = gcv_gradientBoosting.best_estimator_
    gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_

    index = 0
    result_gb = list(gradientBoosting_best.predict(test_X))
    for i in range(len(result_gb)):
       if result_gb[i] != 'Normal':
           result_gb[i] = 'Attack'

    # Confusion Matrix & ROC curve
    print(confusion_matrix(test_y, result_gb))
    print(classification_report(test_y, result_gb, target_names=['Attack', 'Normal']))

    prob_gb = gradientBoosting_best.predict_proba(test_X)
    rocvis(test_y, prob_gb[:, 1], "GradientBoost")

    # best model & score
    # score = [gcv_decisionTree_score, gcv_randomForest_score]
    # models = [decisionTree_best, randomForest_best]
    # model_names = ["Decision Tree", "Random Forest"]
    # model_results = [result_dt, result_rf]
    model_names = ["Decision Tree", "Random Forest","Gradient Boosting"]
    score = [gcv_decisionTree_score, gcv_randomForest_score, gcv_gradientBoosting_score]
    models = [decisionTree_best, randomForest_best, gradientBoosting_best]
    model_results = [result_lr, result_dt, result_rf, result_gb]

    max_score = gcv_logistic_score
    best_model = logistic_best
    best_name = "Logistic Regression"
    best_result = result_lr

    for i in range(len(score)):
        if max_score < score[i]:
            max_score = score[i]
            best_model = models[i]
            best_name = model_names[i]
            best_result = model_results[i]

    print("Anomaly detection best score: ", max_score)

    f = open('misuse_model.txt', 'w')

    result = "Best model  ---------  " + best_name + "\n" + 'best score' + str(
        max_score) +"\n"+ str(confusion_matrix(test_y, best_result)) +"\n"+ str(
        classification_report(test_y, best_result, target_names=['Attack',
                                                                   'Normal'])) + '\n----------------------------------------------------------------------------------------------------\n'
    f.write(result)


    print("Misuse detection best score: ", max_score)

    # a new dataframe by collecting only normal results through best model
    normal_X = pd.DataFrame(columns=test_df.columns)
    normal_y = []
    index = 0
    result = list(best_model.predict(test_X))
    f =  open('misuse.txt', 'w')
    for i in range(len(result)):
        if result[i] == 'Normal':
            attacks['Normal'] = attacks.get('Normal') + 1
            normal_X.loc[index] = test_X[i]
            normal_y.append(test_y[i])
            index = index + 1
        else:
            # check
            # print('index', i, '---->', result[i])
            misuse_result = 'index' + str(i) + '---->' + result[i] + "\n"
            f.write(misuse_result)
            for j in attacks.keys():
                if (j == result[i]):
                    # check
                    # print(result[i], j, sep='       ')
                    attacks[j] = attacks.get(j) + 1
            result[i] = 'Attack'
    print(misuse_result)
    f.close()
    # check
    for j in attacks.keys():
        print(j, ':', attacks[j])
        print()
    normal_y = np.array(normal_y)

    # Draw ROC curve

    plt.plot([0, 1], [0, 1], linestyle='--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # plt.title("Misuse detection Roc Curve")
    plt.savefig('./misuse_ROC.png')

    #Draw pie chart
    plt.cla()
    labels = attacks.keys()
    ratio = attacks.values()
    plt.pie(ratio, labels=labels, shadow=True, startangle=90)
    plt.savefig('./misuse.png')
    # plt.show()

    # Confusion Matrix & ROC curve
    print(confusion_matrix(test_y, result))
    print(classification_report(test_y, result, target_names=['Attack', 'Normal']))

    print(normal_X.columns)
    print(normal_y)

    # ----------------------------------------------------------------------------------------
    print("Anomaly detection")
    plt.cla()
    value = anomaly_x.columns
    X_scaled = MinMax_scaler.fit_transform(anomaly_x)

    # misuse dectection model train
    logistic = LogisticRegression().fit(X_scaled, anomaly_y)
    decisionTree = DecisionTreeClassifier().fit(X_scaled, anomaly_y)
    randomForest = RandomForestClassifier().fit(X_scaled, anomaly_y)
    gradientBoosting = GradientBoostingClassifier().fit(X_scaled, anomaly_y)

    # GridSearch
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # Logistic Regression
    gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_logistic.fit(X_scaled, anomaly_y)
    print("---------------------------------------------------------------")
    print("Logistic Regression")
    print('final params', gcv_logistic.best_params_)  # print best parameter value
    print('best score', gcv_logistic.best_score_)  # best score
    logistic_best = gcv_logistic.best_estimator_
    gcv_logistic_score = gcv_logistic.best_score_

    index = 0
    result_lr = list(logistic_best.predict(normal_X))
    # Confusion Matrix & ROC curve
    print(confusion_matrix(normal_y, result_lr))
    print(classification_report(normal_y, result_lr, target_names=['Attack', 'Normal']))



    prob_lr = logistic_best.predict_proba(test_X)
    rocvis(test_y, prob_lr[:, 1], "Logistic")

    # Decision Tree
    gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_decisionTree.fit(X_scaled, anomaly_y)
    print("---------------------------------------------------------------")
    print("Decision Tree")
    print('final params', gcv_decisionTree.best_params_)  # print best parameter value
    print('best score', gcv_decisionTree.best_score_)  # best score
    decisionTree_best = gcv_decisionTree.best_estimator_
    gcv_decisionTree_score = gcv_decisionTree.best_score_

    index = 0
    result_dt = list(decisionTree_best.predict(normal_X))
    # Confusion Matrix & ROC curve
    print(confusion_matrix(normal_y, result_dt))
    print(classification_report(normal_y, result_dt, target_names=['Attack', 'Normal']))

    prob_dt = decisionTree_best.predict_proba(test_X)
    rocvis(test_y, prob_dt[:, 1], "DecisionTree")

    # Random Forest
    gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_randomForest.fit(X_scaled, anomaly_y)
    print("---------------------------------------------------------------")
    print("Random Forest")
    print('final params', gcv_randomForest.best_params_)  # print best parameter value
    print('best score', gcv_randomForest.best_score_)  # best score
    randomForest_best = gcv_randomForest.best_estimator_
    gcv_randomForest_score = gcv_randomForest.best_score_

    index = 0
    result_rf = list(randomForest_best.predict(normal_X))
    # Confusion Matrix & ROC curve
    print(confusion_matrix(normal_y, result_rf))
    print(classification_report(normal_y, result_rf, target_names=['Attack', 'Normal']))


    prob_rf = randomForest_best.predict_proba(test_X)
    rocvis(test_y, prob_rf[:, 1], "RandomForest")

    # Gradient Boosting
    gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
    gcv_gradientBoosting.fit(X_scaled, anomaly_y)
    print("---------------------------------------------------------------")
    print("Gradient Boosting")
    print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
    print('best score', gcv_gradientBoosting.best_score_)  # best score
    gradientBoosting_best = gcv_gradientBoosting.best_estimator_
    gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_

    index = 0
    result_gb = list(gradientBoosting_best.predict(normal_X))
    # Confusion Matrix & ROC curve
    print(confusion_matrix(normal_y, result_gb))
    print(classification_report(normal_y, result_gb, target_names=['Attack', 'Normal']))


    prob_gb = gradientBoosting_best.predict_proba(test_X)
    rocvis(test_y, prob_gb[:, 1], "GradientBoost")

    # best model & score
    # score = [gcv_decisionTree_score, gcv_randomForest_score]
    # models = [decisionTree_best, randomForest_best]
    # model_names = ["Decision Tree", "Random Forest"]
    # model_results = [result_dt, result_rf]
    model_names = ["Decision Tree", "Random Forest","Gradient Boosting"]
    score = [gcv_decisionTree_score, gcv_randomForest_score, gcv_gradientBoosting_score]
    models = [decisionTree_best, randomForest_best, gradientBoosting_best]
    model_results = [result_lr, result_dt, result_rf, result_gb]

    max_score = gcv_logistic_score
    best_model = logistic_best
    best_name = "Logistic Regression"
    best_result = result_lr
    for i in range(len(score)):
        if max_score < score[i]:
            max_score = score[i]
            best_model = models[i]
            best_name = model_names[i]
            best_result = model_results[i]

    print("Anomaly detection best score: ", max_score)

    f = open('anomaly_model.txt', 'w')

    result = "Best model  ---------  " +best_name +  "\n" + 'best score' + str(
        max_score) +"\n"+ str(confusion_matrix(normal_y, best_result)) +"\n"+ str(
        classification_report(normal_y, best_result, target_names=['Attack',
                                                                 'Normal'])) + '\n----------------------------------------------------------------------------------------------------\n'
    f.write(result)

    f.close()

    result = list(best_model.predict(normal_X))
    with open('anomaly.txt', 'w') as f:
        for i in range(len(result)):
            if result[i] != 'Normal':
                attack_num = attack_num + 1
                # check
                # print('index', i, '---->', result[i])
                anomaly_result = 'index' + str(i) + '---->' + result[i] + '\n'
                f.write(anomaly_result)
                # check
                # print(normal_X.iloc[i])
            else:
                normal_num = normal_num + 1
                # check
                # print('normal_index', i, '---->', result[i])

    print("attack_num : ",attack_num,"normal : ",normal_num)

    # check
    # for i in range(len(normal_y)):
    #     if normal_y[i] != 'Normal':
    #         print('index', i, normal_y[i], '---->', result[i])
    # print(normal_X.iloc[i])

    # check
    # Confusion Matrix & ROC curve
    # print(confusion_matrix(normal_y, result))
    # print(classification_report(normal_y, result, target_names=['Attack', 'Normal']))



    # Draw ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # plt.title("Anomaly detection Roc Curve")
    plt.savefig('./anomaly_ROC.png')

    # Draw pie chart
    plt.cla()
    anomaly_labels = ['Attack', 'Normal']
    anomaly_ratio = [attack_num, normal_num]
    plt.pie(anomaly_ratio, labels=anomaly_labels, shadow=True, startangle=90)
    plt.savefig('./anomaly.png')
    # plt.show()

    return 0