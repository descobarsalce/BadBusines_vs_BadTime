import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, vstack, csr_matrix

# Auxiliary Functions

def open_csv_data(filename):
    '''
    Checks if the file exists and turns it into a pandas dataframe.
    Input: filename (string)
    Output: Pandas df (if file exists)
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
           varlist (list of string with variable names)
    Output: figure
    '''
    corr = df[varlist].corr()
    figure = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    return figure

def crosstab_plt(df, var1, var2, outcome_var, fig_size=(10,5)):
    '''
    Creates a graph of the cross-tabulation with frequencies for each bin.
    Input: df (pandas df)
           var1, var2 (string - variable names)
           outcome_var (string - outcome variable name)
           fig_size (tuple - figure dimension)
    '''
    crossing = pd.crosstab(df[var1], df[var2], values=df[outcome_var], aggfunc=[np.mean])
    figure = plt.figure(figsize=fig_size)
    sns.heatmap(crossing)
    return figure

def cross_section_split(df, test_size=0.33, random_state=1, train_test='train_sample'):
    '''
    Splits the data into test and train and then normalize it.
    Input: df, proportion desired in the test sample, and seed.
    Output: variable identifying observations to be used in the training sample.
    '''
    np.random.seed(seed=random_state)
    df[train_test] = np.random.binomial(n=1, p=test_size, size=len(df))
    df[train_test] = np.where(df[train_test]==1, "train", "test")
    return df

def sort_continuous_categorical(df, omit_list, threshold, train_test='train_sample'):
    '''
    Sorts variables into continuous and categorical variables based on the number of different values they take.
    Input: df
           threshold (int) of how many unique values to consider a variable categorical
    Output: list of continuous variables, list of categorical variables.
    '''
    df = df[df[train_test]=="train"]
    continuous = []
    categorical = []
    not_usable = []  # Defined arbitrarily as non-continuous variables that take more values than what is specified by the threshold.
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

# Data preprocessing

def pre_processing(df, continuous, categorical, outcome_var, train_test='train_sample'):
    '''
    Does the processing steps of creating dummies, cleaning outliers, replacing nulls, and discretizing continuous variables.
    Input: df (pandas dataframe)
           continuous (set of strings - continuous variable names)
           categorical (set of strings - categorical variable names)
           outcome_var (string - outcome variable name)
           train_test (string - train/test identifier)
    Output: x_train, y_train, x_test, y_test (numpy arrays)
    '''
    df = clean_outliers(df, continuous, 4, train_test)
    df = missings_to_mean(df, continuous, train_test)
    df, var_quantiles = gen_quantiles(df, continuous, 5, train_test)
    categorical = list(categorical) + var_quantiles
    x_train, y_train, x_test, y_test = data_format(df, continuous, categorical, outcome_var, train_test)
    return x_train, y_train, x_test, y_test

def clean_outliers(df, var_list, max_std=4, train_test='train_sample'):
    '''
    Replaces values above a certain threshold with the replacement value.
    Input: df (pandas dataframe)
           var_list (list of strings - variable names)
           max_std (float) - number of standard deviations to be used as maximum for outliers
           train_test (string - train/test identifier)
    Output: cleaned dataframe
    '''
    for var in var_list:
        max_val = max_std * df[df[train_test] == "train"][var].std()
        df[var] = np.where(df[var] <= max_val, df[var], max_val)
    return df

def missings_to_mean(df, variable_list, train_test='train_sample'):
    '''
    Replaces null values with the mean.
    Input: df (pandas dataframe)
           variable_list (list of strings - variable names)
           train_test (string - train/test identifier)
    Output: cleaned dataframe
    '''
    for var in variable_list:
        df[var] = df[var].fillna(df[df[train_test] == "train"][var].mean())
    return df

def gen_quantiles(df, var_list, n_bins=3, train_test='train_sample'):
    '''
    Divides the data into n_bins quantiles.
    Input: df (pandas dataframe)
           var_list (list of strings - variable names)
           n_bins (int) - number of quantiles to split the data into
           train_test (string - train/test identifier)
    Output: dataframe with new quantile variables, list of newly created variable names
    '''
    new_vars = []
    for var in var_list:
        new_var = var + "_bins"
        lower_bound = df[var][df[train_test] == "train"].min()
        df[new_var] = 0
        for q in range(1, n_bins + 1):
            upper_bound = df[var][df[train_test] == "train"].quantile(q / n_bins)
            df.loc[(lower_bound <= df[var]) & (df[var] <= upper_bound), new_var] = q
            lower_bound = upper_bound
        new_vars.append(new_var)
    return df, new_vars

def gen_dummies(df, var_list, train_test='train_sample'):
    '''
    Create dummies from specific variables.
    Input: df (pandas dataframe)
           var_list (list of strings - variable names)
           train_test (string - train/test identifier)
    Output: x_train, x_test (numpy arrays)
    '''
    for var in var_list:
        df[var] = df[var].astype(str).fillna('missing_values_pipeline_added')

    s = OneHotEncoder(handle_unknown='ignore')
    s.fit(df[df[train_test] == "train"][var_list])
    x_train = s.transform(df[df[train_test] == "train"][list(var_list)])
    x_test = s.transform(df[df[train_test] == "test"][list(var_list)])

    return x_train, x_test

def data_format(df, continuous, categorical, outcome_var, train_test='train_sample'):
    '''
    Transform pandas dataframe to numpy arrays with float32 data type required by sklearn.
    Input: df (pandas dataframe)
           continuous (set of strings - continuous variable names)
           categorical (set of strings - categorical variable names)
           outcome_var (string - outcome variable name)
           train_test (string - train/test identifier)
    Output: x_train, y_train, x_test, y_test (numpy arrays)
    '''
    x_train, x_test = gen_dummies(df, categorical, train_test)

    if continuous:
        min_max_scaler = MinMaxScaler()
        x_train_cont = df[list(continuous)][df[train_test] == "train"]
        x_test_cont = df[list(continuous)][df[train_test] == "test"]
        min_max_scaler.fit(x_train_cont)
        x_train_cont = min_max_scaler.transform(x_train_cont)
        x_test_cont = min_max_scaler.transform(x_test_cont)
        x_train = hstack((vstack(csr_matrix(x_train_cont)), vstack(x_train)))
        x_test = hstack((vstack(csr_matrix(x_test_cont)), vstack(x_test)))

    y_train = np.array(df[outcome_var][df[train_test] == "train"], dtype=np.float32).ravel()
    y_test = np.array(df[outcome_var][df[train_test] == "test"], dtype=np.float32).ravel()
    return x_train, y_train, x_test, y_test

def scale_data(x_train, x_test):
    '''
    Computes the scaling using the train data and applies it to the test data.
    Input: x_train, x_test (numpy arrays)
    Output: scaled x_train, x_test (numpy arrays)
    '''
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test
