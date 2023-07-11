'''
Homework  Diego Escobar
May 30th 2019
'''

import pipeline as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from dateutil.relativedelta import relativedelta
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


class ModelML:
    '''
    Sklearn model stored together with its estimation result and parameters.
    Input: model type and split data.
    '''

    def __init__(self, model, df, outcome_var, train_test='train_sample', date_var='', min_train_months=6, test_length=3, omit_list=[], leave_out_months=2):
        self.model = classifiers[model]
        self.df = df
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.pr_test_hat = 0

        # Database parameters (to recognize what variable represents what):
        self.outcome_var = outcome_var
        self.train_test = train_test
        self.date_var = date_var
        self.min_train_months = min_train_months
        self.test_length = test_length
        self.leave_out_months = leave_out_months
        self.omit_list = omit_list
        self.prec_rec = {'precision': 0, 'recall': 0, 'roc_auc': 0, 'F1': 0}

    def crossval_rolling_window(self, k=0.05, _print=False, **params):
        '''
        Computes cross-validation using a rolling window and returns average stats.
        '''
        # Split using temporal holdout:
        samples_set = self.cv_date_ranges(min_test_date=pd.to_datetime('2007-01-01'))
        # Fit the models for each sample of train/test data:
        cv_models = self.run_temp_holdout_cv(samples_set, _print, **params)
        # Compute desired stats as average over samples:
        avg_at_k = self.get_CVRW_average_prec_rec_at_k(cv_models, k=k)
        return avg_at_k

    def cv_date_ranges(self, min_test_date):
        '''
        Find the date ranges to be used in rolling window cross-validation. It will
        store the ranges rather than separate databases to save memory.
        '''
        self.df = self.df.sort_values(by=[self.date_var])
        
        # Find the beginning of the first test subsample.
        start_date = min_test_date
        last_date = self.df[self.date_var].max()
        date_ranges = []
        date_begin = start_date
        while date_begin < last_date:
            date_end = date_begin + relativedelta(months=+self.test_length)
            date_ranges.append((date_begin, date_end))
            date_begin = date_begin + relativedelta(months=+12)
        return date_ranges

    def run_temp_holdout_cv(self, samples_set, _print=False, **params):
        '''
        Gets test and training data out of date ranges defined by rolling
        window cross-validation.
        '''
        cv_models = []
        for dates in samples_set:
            self.df[self.train_test] = np.where(self.df[self.date_var] <= dates[0], "train", "")
            self.df[self.train_test] = np.where((dates[0] < self.df[self.date_var]) & (self.df[self.date_var] <= dates[1]), "test", self.df[self.train_test])
            self.df[self.train_test] = np.where((self.df[self.train_test]=="train") & (self.df[self.date_var] >= dates[0] + relativedelta(months=-self.leave_out_months)), "out", self.df[self.train_test])
            if _print:
                print(self.df[self.train_test].value_counts(sort=False))

            # We have to do all the preprocessing for each fold of the sample.
            # This may take longer but takes all the info available until that time, including, for instance, new values for categorical variables.
            continuous, categorical, not_usable = pl.sort_continuous_categorical(df=self.df, omit_list=self.omit_list, threshold=100, train_test=self.train_test)
            x_train, y_train, x_test, y_test = pl.pre_processing(self.df, continuous, categorical, self.outcome_var)
            self.change_sample(x_train, x_test, y_train, y_test)
            self.fit_model(**params)
            
            cv_models.append((self.y_test, self.pr_test_hat))
            
        return cv_models
    
    def change_sample(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def fit_model(self, **parameters):
        '''
        Fit the model to the data.
        Input: potential parameters to be set. If not specified, the
                default version of the model is run.
        '''
        self.model.set_params(**parameters)
        self.model.fit(self.x_train, self.y_train)
        self.pr_test_hat = np.transpose(self.model.predict_proba(self.x_test))[1]
        results = {'model': self.model, 'pr_test_hat': self.pr_test_hat}
        return results

    def get_CVRW_average_prec_rec_at_k(self, cv_models, k):
        '''
        Computes average precision/recall over models.
        '''
        avg_at_k = {'precision': 0.0, 'recall': 0.0, 'roc_auc': 0.0, 'F1': 0}
        for y_hat_true in cv_models:   
            y_true = y_hat_true[0]
            pred_y_prob = y_hat_true[1]
            prec_rec = self.prec_recall_at_k(y_true, pred_y_prob, k)
            for key, value in prec_rec.items():
                avg_at_k[key] += value / len(cv_models)
        return avg_at_k

    def gen_data_samples(self, samples_set, _print=False):
        '''
        Gets test and training data out of date ranges defined by rolling
        window cross-validation.
        '''
        data = []
        for dates in samples_set:  
            self.df.loc[self.df[self.date_var] <= dates[0], self.train_test] =  "train"
            self.df.loc[(dates[0] < self.df[self.date_var]) & (self.df[self.date_var] <= dates[1]), self.train_test] = "test"
            if _print:
                print(self.df[self.train_test].value_counts(sort=False))
            # We have to do all the preprocessing for each fold of the sample.
            # This may take longer but takes all the info available until that time, including, for instance, new values for categorical variables.
            continuous, categorical, not_usable = pl.sort_continuous_categorical(df=self.df, omit_list=self.omit_list, threshold=100, train_test=self.train_test)
            x_train, y_train, x_test, y_test = pl.pre_processing(self.df, continuous, categorical, self.outcome_var)            
            x_train = pd.DataFrame(x_train.toarray())
            y_train = pd.DataFrame(y_train)
            x_test = pd.DataFrame(x_test.toarray())
            y_test = pd.DataFrame(y_test)
            data.append([x_train, y_train, x_test, y_test])
        return data
    
    def plot_precision_recall_n(self, graph_name):
        '''
        Description: Creates the precision-recall graph for a model.
        Inputs:
            y_true: Real Labels
            y_prob: Predicted Labels.
            model_name: Some fitted model.
        Output:
            Precision-Recall graph.
        '''
        y_true = self.y_test
        y_score = self.pr_test_hat
        
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)
        
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0, 1])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, 1])
        
        plt.title(graph_name)
        plt.show()
    
    def prec_recall_at_k(self, y_true, pred_y_prob, k):
        '''
        Computes precision and recall taking the first k as positives.
        '''    
        pred_y_prob, y_true_sorted = joint_sort_descending(np.array(pred_y_prob), np.array(y_true))
        y_pred_k = generate_binary_at_k(pred_y_prob, k)        

        prec_rec = {
            'precision': metrics.precision_score(y_true_sorted, y_pred_k),
            'recall': metrics.recall_score(y_true_sorted, y_pred_k),
            'roc_auc': metrics.roc_auc_score(y_true_sorted, y_pred_k),
            'F1': metrics.f1_score(y_true_sorted, y_pred_k)
        }
        self.prec_rec = prec_rec
        return prec_rec
    
    def scoring_measures_thresholds(self,
            k_values=np.linspace(0.01, 0.5, 10, endpoint=False)):
        '''
        Computes precision, recall, and auc curve for a given model using
        different thresholds.
        '''
        precision = []
        recall = []
        roc_auc = []
        f1 = []
        for k in k_values:
            y_true, y_prob_pred = self.y_test, self.pr_test_hat
            
            prec_rec = self.prec_recall_at_k(y_true, y_prob_pred, k)
            
            precision.append(prec_rec['precision'])
            recall.append(prec_rec['recall'])
            roc_auc.append(prec_rec['roc_auc'])
            f1.append(prec_rec['F1'])
        return precision, recall, roc_auc, f1, k_values

    def roc_curve(self, name,
                  k_values=np.linspace(0.01, 0.5, 10, endpoint=False)):
        '''
        Plots precision, recall, and auc curve for a given model using different
        thresholds.
        Input: model name to display in the graph title and a thresholds list.
        '''
        prec, recall, roc_auc, f1, z = self.scoring_measures_thresholds(k_values)
        # ROC figure
        fig, ax = plt.subplots(1)
        # Plot precision and recall vs x in blue on the left vertical axis.
        plt.xlabel("Threshold")
        plt.ylabel("AUC ROC", color="b")
        plt.tick_params(axis="y", labelcolor="b")
        plt.plot(k_values, roc_auc, "b-", linewidth=1)
        plt.title("AUC ROC: " + name)
        fig.autofmt_xdate(rotation=50)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The following functions are independent and not part of the Model_ML class:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generate_binary_at_k(y_scores, k):
    '''
    Generate binary values at a specified threshold level.
    Inputs:
        y_scores: Predicted scores
        k: Threshold level
    Output:
        Binary predictions at the threshold level
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def metric_at_k(y_true, y_scores, k, mtrc):
    '''
    Calculate a specified metric at a specified threshold level.
    Inputs: 
        y_true: True labels to predict
        y_scores: Predicted scores
        k: Threshold level
        mtrc: Metric function to calculate
    Output:
        Metric value at the threshold level
    '''
    y_scores, y_true = joint_sort_descending(y_scores, y_true)
    preds_at_k = generate_binary_at_k(y_scores, k)
    return mtrc(y_true, preds_at_k)

def precision_at_k(y_true, y_scores, k):
    '''
    Calculate precision at a specified threshold level.
    Inputs: 
        y_true: True labels to predict
        y_scores: Predicted scores
        k: Threshold level
    Output:
        Precision value at the threshold level
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    Calculate recall at a specified threshold level.
    Inputs:
        y_true: True labels to predict
        y_scores: Predicted scores
        k: Threshold level
    Output:
        Recall value at the threshold level
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)
    return recall

def f1_at_k(y_true, y_scores, k):
    '''
    Calculate F1 score at a specified threshold level.
    Inputs:
        y_true: True labels to predict
        y_scores: Predicted scores
        k: Threshold level
    Output:
        F1 score at the threshold level
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    f1 = f1_score(y_true, preds_at_k)
    return f1

def accuracy_at_k(y_true, y_scores, k):
    '''
    Calculate accuracy at a specified threshold level.
    Inputs:
        y_true: True labels to predict
        y_scores: Predicted scores
        k: Threshold level
    Output:
        Accuracy at the threshold level
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    accuracy = accuracy_score(y_true, preds_at_k)
    return accuracy

def joint_sort_descending(l1, l2):
    '''
    Jointly sort two vectors in descending order based on the first vector.
    Inputs:
        l1: Vector for sorting
        l2: Vector to be sorted
    Output:
        Sorted vectors based on the first vector
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]
    
def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, levels=np.linspace(1.0, 100, 10, endpoint=True)):
    """
    Run the loop using specified models, classifiers, parameter grid, and data.
    """
    param_list = ["p_at_", "r_at_", "f_at_", "acc_at_"]
    base_param = ['model_type', 'clf', 'parameters', 'auc-roc']
    
    for j in param_list:
        for i in levels:
            part_str = j + str(i)
            base_param.append(part_str)
    
    results_loop_df =  pd.DataFrame(columns=base_param)
    
    for n in range(1, 2):
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            parameter_values = grid[models_to_run[index]]
            
            for p in ParameterGrid(parameter_values):
                model = clf
                model.set_params(**p)
                model.fit(x_train.values, y_train.values.ravel())
                
                y_pred_probs = model.predict_proba(x_test)[:, 1]
                y_test = np.array(y_test)
                y_pred_probs = np.array(y_pred_probs)
                
                y_pred_probs_sorted, y_test_sorted = joint_sort_descending(y_pred_probs, y_test)
                base_columns = [models_to_run[index], model, p, roc_auc_score(y_test_sorted, y_pred_probs_sorted)]
                
                for i in levels:
                    precision = precision_at_k(y_test_sorted, y_pred_probs_sorted, i)
                    base_columns.append(precision)
                
                for i in levels:
                    recall = recall_at_k(y_test_sorted, y_pred_probs_sorted, i)
                    base_columns.append(recall)
                
                for i in levels:
                    f1 = f1_at_k(y_test_sorted, y_pred_probs_sorted, i)
                    base_columns.append(f1)
                
                for i in levels:
                    accuracy = accuracy_at_k(y_test_sorted, y_pred_probs_sorted, i)
                    base_columns.append(accuracy)

                results_loop_df.loc[len(results_loop_df)] = base_columns
    
    return results_loop_df

def model_selector_corr(data, param_grid, clfs, t, models_to_run, levels=np.linspace(1.0, 100, 10, endpoint=True), gap=0):
    '''
    Calculate clf_loop for rolling window.
    Inputs:
        data: Data samples
        param_grid: Parameter grid
        clfs: Preloaded grid of models to run
        t: Time variable in timestamp
        models_to_run: List of selected models to run with some initial parameters.
        levels: List of threshold levels
    Output:
        Matrix of results
    '''
    param_list = ["p_at_", "r_at_", "f_at_", "acc_at_"]
    base_param = ['test_data', 'model_type', 'clf', 'parameters', 'auc-roc']

    for j in param_list:
        for i in levels:
            part_str = j + str(i)
            base_param.append(part_str)
    
    results_df = pd.DataFrame(columns=base_param)
    
    for i, dt in enumerate(data):
        x_train, y_train, x_test, y_test = dt
        result = clf_loop(models_to_run, clfs, param_grid, x_train, x_test, y_train, y_test, levels)
        results_df = results_df.append(result)
        results_df['test_data'] = results_df['test_data'].fillna('test' + str(i+1)) 
    
    return results_df[base_param]

# All the classifiers to be tested```python
classifiers = {
    'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=200),
    'LR': LogisticRegression(penalty='l1', C=1e5, max_iter=1000),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=0, max_iter=250),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(max_depth=5),
    'SGD': SGDClassifier(loss='log', penalty='l2'),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

# Parameters to be run through
large_grid = {
    'RF': {'n_estimators': [1, 10, 100, 1000, 10000], 'max_depth': [1, 5, 10, 20, 50, 100],
           'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10], 'n_jobs': [-1]},
    'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
    'SGD': {'loss': ['hinge', 'log', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet']},
    'ET': {'n_estimators': [1, 10, 100, 1000, 10000], 'criterion': ['gini', 'entropy'],
           'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10],
           'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1, 10, 100, 1000, 10000]},
    'GB': {'n_estimators': [1, 10, 100, 1000, 10000], 'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
           'subsample': [0.1, 0.5, 1.0], 'max_depth': [1, 3, 5, 10, 20, 50, 100]},
    'NB': {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100],
           'max_features': [None, 'sqrt', 'log2'], 'min_samples_split': [2, 5, 10]},
    'SVM': {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
    'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']}
}

small_grid = {
    'RF': {'n_estimators': [100, 200], 'max_depth': [10, 20, 40], 'min_samples_split': [10], 'n_jobs': [-1]},
    'LR': {'penalty': ['l2'], 'C': [0.001, 1, 100]},
    'DT': {'criterion': ['gini'], 'max_depth': [20, 50, 100, 200], 'min_samples_split': [2, 5, 10]},
    'GB': {'n_estimators': [100], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [10, 20]},
    'SGD': {'loss': ['log'], 'penalty': ['l2', 'l1', 'elasticnet']},
    'ET': {'n_estimators': [100, 200], 'criterion': ['gini'], 'max_depth': [5, 50],
           'max_features': ['sqrt'], 'min_samples_split': [2, 10], 'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME'], 'n_estimators': [1, 50]},
    'NB': {},
    'SVM': {'C': [1, 10], 'kernel': ['linear']},
    'KNN': {'n_neighbors': [5, 25, 100], 'weights': ['uniform'], 'algorithm': ['auto']}
}

test_grid = {
    'RF': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10], 'n_jobs': [-1]},
    'LR': {'penalty': ['l2'], 'C': [0.01]},
    'SGD': {'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': {'n_estimators': [1], 'criterion': ['gini'], 'max_depth': [1],
           'max_features': ['sqrt'], 'min_samples_split': [10], 'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate': [0.1], 'subsample': [0.5], 'max_depth': [1]},
    'NB': {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': [None], 'min_samples_split': [10]},
    'SVM': {'C': [0.01], 'kernel': ['linear']},
    'KNN': {'n_neighbors': [5], 'weights': ['uniform'], 'algorithm': ['auto']}
}
