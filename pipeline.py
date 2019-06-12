'''
Homework  Diego Escobar
May 30th 2019
'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, vstack, csr_matrix

# Auxiliary Functions:
def open_csv_data(filename):
    '''
    Checks if the file exists and turns it into a pandas dataframe.
    Input: filename (string)
    Output: Pandas df (if file exist)
    '''
    try:
        df = pd.read_csv(filename)
        return df
    except (FileNotFoundError, IOError):
        raise NameError('Error: File Not Found.')

def correlation_plot(df, varlist):
    '''
    Generates a plot displaying the correlation matrix.
    Input: dataframe (pandas)
           varlist (list of string with var names)
    Output: figure
    '''
    corr = df[varlist].corr()
    figure = sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    return figure

def crosstab_plt(df, var1, var2, outcome_var, fig_size=(10,5)):
    '''
    Creates a graph of the cross-tabulation with frequencies for each bin.
        Input: df (pandas df)
               var1, var2 (string - var names)
               _norm (boolean - whether to normalize frequency or not).
               fig_size (tuple - figure dimension).
    '''
    crossing = pd.crosstab(df[var1], df[var2], values=df[outcome_var], \
                            aggfunc=[np.mean])
    figure = plt.figure(figsize=fig_size)
    sns.heatmap(crossing)
    return figure

def cross_section_split(df, test_size=0.33, random_state=1,  train_test='train_sample'):
    '''
    Splits the data into test and train and then normalize it.
    Input: df, proportion desired in the test sample, and seed.
    Output: variable identiifying observations to be used in the training sample.
    '''
    np.random.seed(seed=random_state)
    df[train_test] = np.random.binomial(n=1, p=test_size, size=len(df))
    df[train_test] = np.where(df[train_test]==1, "train", "test")
    return df

def sort_continuous_categorical(df, omit_list, threshold, train_test='train_sample'):
    '''
    Sorts variables into continuous and categorical variables based on the
    number of different values they take.
    Input: df
        threshold (int) of how many unique values to consider a var categorical
    Output: list of continuous vars, list of categorical vars.
    '''
    df = df[df[train_test]=="train"]
    continuous = []
    categorical = []
    not_usable = [] # Defined arbitrarily as non continuous variables that take
    # more values than what is specified by threshold.
    all_vars = set(list(df)) - set(omit_list)
    for var in all_vars:
        if df[var].nunique() > threshold:
            if np.issubdtype(df[var].dtype, np.number):
                continuous.append(var)
            else:
                not_usable.append(var)
        else:
            categorical.append(var)
    return set(continuous), set(categorical), set(not_usable)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#Data preprocessing:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This has to be done for each training sample.

def pre_processing(df, continuous, categorical, outcome_var, train_test='train_sample'):
    '''
    Does the processing steps of creating dumm
    '''
    # For the missing values I will either replace with zero and create a dummy
    # indicating that (statistically more meaninful in my opinion) or replace with
    # mean or median:

    df = clean_outliers(df, continuous, 4, train_test)
    df = missings_to_mean(df, continuous, train_test)

    # Discretize continuous variables:
    df, var_quantiles = gen_quantiles(df, continuous, 5, train_test)
    categorical = list(categorical) + var_quantiles

    x_train, y_train, x_test, y_test = data_format(df, continuous, categorical, outcome_var, train_test)

    return x_train, y_train, x_test, y_test

# Fixing outliers:
def clean_outliers(df, var_list, max_std=4, train_test='train_sample'):
    '''
    Replaces values above certain threshold with the replacement value.
    input: dataframe (pandas), variable name (string), max_std (float) is the
    number of standard deviations to be used as maximum for outliers.
    '''
    for var in var_list:
        max_val = max_std*df[df[train_test]=="train"][var].std()
        df[var] = np.where(df[var] <= max_val, df[var], max_val)
    return df

# Replaces nulls with mean or median:
def missings_to_mean(df, variable_list, train_test='train_sample'):
    '''
    Replaces null values with mean
    Input: dframe (pandas df)
           variable to clean (list of strings)
    Output: Changes are made in the dataset
    '''
    for var in variable_list:
        df[var] = df[var].fillna(df[df[train_test]=="train"][var].mean())
    return df

def gen_quantiles(df, var_list, n_bins=3, train_test='train_sample'):
    '''
    Divides the data into n_bins-quantiles.
    Input: n_bins (int -> in how many quantiles to split the data)
           labels (list of string -> names for quantiles, optional)
    '''
    new_vars = []
    for var in var_list:
        new_vars.append(var + "_bins")
        lower_bound = df[var][df[train_test]=="train"].min()
        df.loc[:, var + '_bins'] = 0
        for q in range(1 , n_bins + 1):
            upper_bound = df[var][df[train_test]=="train"].quantile(q/n_bins)
            df.loc[(lower_bound <= df[var]) & (df[var] <= upper_bound), var + '_bins'] = q
            #df.loc[not ((lower_bound <= df[var]) & (df[var] <= upper_bound)), var + '_bins'] = df[var + '_bins']
            lower_bound = upper_bound
    return df, new_vars

def gen_dummies(df, var_list, train_test='train_sample'):
    '''
    Create dummies from specific variables.
    Input: df (pandas dataframe)
           var_list (list of strings)
    Output: name of newly created variables (list of strings)
    '''
    # First I'll replace nulls so that dummies are created for them as well.
    for var in var_list:
        df[var] = df[var].astype(str).fillna('missing_values_pipeline_added')
    '''
    for var in var_list:
        if np.issubdtype(df[var].dtype, np.number):
            df[var] = df[var].fillna(df[var].max() + 1)
        else:
            df[var] = df[var].fillna('missing_values_pipeline_added')
    '''
    s = OneHotEncoder(handle_unknown='ignore')
    s.fit(df[df[train_test]=="train"][var_list])
    x_train = s.transform(df[df[train_test]=="train"][list(var_list)])
    x_test = s.transform(df[df[train_test]=="test"][list(var_list)])
    
    return x_train, x_test

def data_format(df, continuous, categorical, outcome_var, train_test='train_sample'):
    '''
    Transform pandas df to np array and data type float32 required by sklearn
    '''
    # Now I will generate dummies for each group of the categorical data:
    x_train, x_test = gen_dummies(df, categorical)

    if continuous:
        min_max_scaler = MinMaxScaler()
        x_train_cont = df[list(continuous)][df[train_test]=="train"]
        x_test_cont  = df[list(continuous)][df[train_test]=="test"]
        min_max_scaler.fit(x_train_cont)
        x_train_cont = min_max_scaler.transform(x_train_cont)
        x_test_cont = min_max_scaler.transform(x_test_cont)
        # Merge this with the continuous var and put everything into scipy matrix format.
        x_train = hstack((vstack(csr_matrix(x_train_cont)), vstack(x_train)))
        x_test = hstack((vstack(csr_matrix(x_test_cont)), vstack(x_test)))
        
    y_train = np.array(df[outcome_var][df[train_test]=="train"],  dtype=np.float32).ravel()
    y_test = np.array(df[outcome_var][df[train_test]=="test"],  dtype=np.float32).ravel()
    return x_train, y_train, x_test, y_test

def scale_data(x_train, x_test):
    '''
    Computes the scaling using train data and applies it to test data.
    '''
    # Some of the methods require normalization (LASSO, NN) so I will normalize
    # First with the training data and then apply the same scale as test data.
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test
