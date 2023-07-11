'''
import autoreload
%load_ext autoreload
%autoreload 2
'''
import pandas as pd
import numpy as np
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pipeline as pl
import pipeline_models as plm

start = time.time()

# Parameters:
filename = "Data/Processed_Data.csv"

# Data loading and spliting:
df = pl.open_csv_data(filename)
df = df[df['duration']>=10] # Discarding firms with less than 10 days (temporary licenses).

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Manual pre-processing for non-standard vars dependent to each dataset.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
We drop variables that we dont want to use such as IDs or names, variables that
include leaked information (e.g. 'duration', that was used when constructing
the outcomes), and some other specific cases. Everything else is considered as
possible features for the model.
'''
drop_vars = ['ACCOUNT NUMBER', 'ID', 'LICENSE ID', 'LICENSE STATUS', 'SITE NUMBER', 'ADDRESS', 'CITY', 'STATE', 'LICENSE NUMBER', 'LATITUDE', 'LONGITUDE', 'LOCATION', 'positive_duration', 'coordinates', 'final_year', '_merge', 'duration', 'index_right', 'Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y', 'APPLICATION CREATED DATE', 'PAYMENT DATE', 'APPLICATION REQUIREMENTS COMPLETE', 'APPLICATION TYPE', 'LICENSE APPROVED FOR ISSUANCE', 'DATE ISSUED', 'LEGAL NAME', 'DOING BUSINESS AS NAME', 'LICENSE STATUS CHANGE DATE', 'LICENSE TERM EXPIRATION DATE_x', 'LICENSE TERM EXPIRATION DATE_y']
df.drop(drop_vars, axis=1, inplace=True)

outcomes = ['less_1_year', '1_2_years', '2_3_years', 'more_3_years']
date_var = 'start_year'
train_test ='train_sample'
test_length = 12
algorithm = 'LR'
omit_list = outcomes + ['duration'] + [date_var, train_test]
outcome_var = outcomes[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions parameters:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

grid_select = 'small_grid'
random_state = 1
min_train_months = 12
test_size = 0.3
threshold = 200
k_grid = [1,2,5,10,20,30,50,70,100]
grid_opts = {'large_grid': plm.large_grid,
             'small_grid': plm.small_grid,
             'test_grid': plm.test_grid,
             'final_grid : finalgrid}
             

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Estimation Parameters:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

outcomes = ['more_3_years']
max_year_outcome = {'less_1_year': pd.Timestamp('2017-01-01 00:00:00'), 
                    '1_2_years': pd.Timestamp('2016-01-01 00:00:00'),
                    '2_3_years': pd.Timestamp('2015-01-01 00:00:00'),
                    'more_3_years': pd.Timestamp('2014-01-01 00:00:00')}

df[date_var] = pd.to_datetime(df[date_var])
df[date_var] = df[date_var].apply(pd.to_datetime)

MAIN_ALL = []
ALL_RESULTS = []
for i, outcome_var in enumerate(outcomes):
    test_length_outcome = 12 # Months included in each test set.
    df_sub = df[df[date_var]<=max_year_outcome[outcome_var]]
    new_model = plm.ModelML(model=algorithm,
                            df=df_sub,
                            outcome_var=outcome_var,
                            date_var=date_var,
                            test_length=test_length_outcome,
                            omit_list=omit_list)
    samples_set = new_model.cv_date_ranges(min_test_date=pd.to_datetime('2008-01-01'))
    data = new_model.gen_data_samples(samples_set, _print=False)
    
    time_short_models = time.time()
    models_to_fit = ['DT', 'LR', 'RF']
    
    results_df = plm.model_selector_corr(data,grid_opts[grid_select],plm.classifiers,0,models_to_fit, levels=k_grid, gap=0)
    
    # This stores the results of ALL models for each set of dates.
    MAIN_ALL.append(results_df)

    '''
    results_df_best = results_df.groupby(['test_data', 'model_type'], as_index=False).max()
    date_df = pd.DataFrame(samples_set, columns={'start_date', 'end_date'})
    for ind in range(0, len(date_df)):
        date_df.loc[ind, 'test_data'] = "test" + str(ind + 1)
    date_df = date_df.merge(results_df_best, left_on='test_data', right_on='test_data')
    '''
    # This stores the results of THE BEST models for each set of dates.
    #ALL_RESULTS.append(date_df)
    
    results_df.to_csv('Data/DIEGO_LR_RF_friday' + str(i) +'.csv')
