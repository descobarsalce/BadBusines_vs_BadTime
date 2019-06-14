'''
import autoreload
%load_ext autoreload
%autoreload 2
'''
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import geopandas
from geopandas import GeoDataFrame
warnings.simplefilter(action='ignore', category=FutureWarning)
 from shapely.geometry import Point
import pipeline as pl
import pipeline_models as plm

start = time.time()

# Parameters:
filename = "Data/Processed_Data.csv"

# Data loading and spliting:
df2 = pd.read_csv("Data/Processed_Data.csv")
df = df[df['duration']>=10] # Discarding firms with less than 10 days (temporary licenses).

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Manual pre-processing for non-standard vars dependent to each dataset.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

drop_vars = ['ACCOUNT NUMBER', 'ID', 'LICENSE ID', 'SITE NUMBER', 'ADDRESS', 'CITY', 'STATE', 'LICENSE NUMBER', 'LATITUDE', 'LONGITUDE', 'LOCATION', 'positive_duration', 'coordinates', 'final_year', '_merge', 'duration', 'index_right', 'Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y', 'APPLICATION CREATED DATE', 'PAYMENT DATE', 'APPLICATION REQUIREMENTS COMPLETE', 'APPLICATION TYPE', 'LICENSE APPROVED FOR ISSUANCE', 'DATE ISSUED', 'LEGAL NAME', 'DOING BUSINESS AS NAME', 'LICENSE STATUS CHANGE DATE', 'LICENSE TERM EXPIRATION DATE_x', 'LICENSE TERM EXPIRATION DATE_y']
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
threshold = 200

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Estimation Parameters:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

outcomes = ['less_1_year', 'more_3_years']
max_year_outcome = {'less_1_year': pd.Timestamp('2017-01-01 00:00:00'), 
                    '1_2_years': pd.Timestamp('2016-01-01 00:00:00'),
                    '2_3_years': pd.Timestamp('2015-01-01 00:00:00'),
                    'more_3_years': pd.Timestamp('2014-01-01 00:00:00')}
df[date_var] = pd.to_datetime(df[date_var])
df[date_var] = df[date_var].apply(pd.to_datetime)
df.sort_values([date_var], inplace=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Predictions Crossing:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Avoiding firms that do not have enough years of data:
df_sub = df[df[date_var]<=max_year_outcome['more_3_years']]

one_year_algorithm = 'RF'
one_year_param = {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 200, 'n_jobs': -1}
model1 = plm.ModelML(model=one_year_algorithm,
                        df=df_sub,
                        outcome_var='less_1_year',
                        date_var=date_var,
                        min_train_months=min_train_months,
                        test_length=test_length,
                        omit_list=omit_list)

three_year_algorithm = 'RF'
three_year_param = {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 200, 'n_jobs': -1}
model3 = plm.ModelML(model=three_year_algorithm,
                        df=df_sub,
                        outcome_var='more_3_years',
                        date_var=date_var,
                        min_train_months=min_train_months,
                        test_length=test_length,
                        omit_list=omit_list)

samples_set = model3.cv_date_ranges(min_test_date=pd.to_datetime('2008-01-01'))

period = len(samples_set)-1 # last year in the data

dates = samples_set[period]
working_x_df = model1.df
working_x_df.loc[working_x_df[model1.date_var] <= dates[0], model1.train_test] =  "train"
working_x_df.loc[(dates[0] < working_x_df[model1.date_var]) & (working_x_df[model1.date_var] <= dates[1]), model1.train_test] = "test"
working_x_df[model1.train_test].value_counts()

data_model = model1.gen_data_samples(samples_set, _print=False)
x_train1, y_train1, x_test1, y_test1 = data_model[period]

data_model = model3.gen_data_samples(samples_set, _print=False)
x_train3, y_train3, x_test3, y_test3 = data_model[period]

model1.change_sample(x_train1, x_test1, y_train1.values.ravel(), y_test1.values.ravel())
results1 = model1.fit_model(**one_year_param)
y_prob1 = np.transpose(model1.model.predict_proba(x_test1))[1]

model3.change_sample(x_train3, x_test3, y_train3.values.ravel(), y_test3.values.ravel())
results3 = model3.fit_model(**three_year_param)
y_prob3 = np.transpose(model3.model.predict_proba(x_test3))[1]

probs_df = pd.DataFrame(np.transpose([y_prob1, y_prob3]))
x_df = working_x_df[working_x_df[model1.train_test]=='test'].reset_index()
full_df = probs_df.join(x_df)

probs_y_test_df = pd.DataFrame(np.transpose([y_prob1]))
probs_y_test_df = probs_y_test_df.merge(y_test1, left_index =True, right_index= True)
full_aequitas_df = probs_y_test_df.join(x_df)


full_df.to_csv('Data/cross_predictions' + str(period) + '.csv')
full_aequitas_df.to_csv('Data/aequitas' + str(period) + '.csv')

model1.plot_precision_recall_n(graph_name='Precision-Recall: One Year Survival')
model3.plot_precision_recall_n(graph_name='Precision-Recall: Three+ Year Survival')

###################################################33

# Plot differences:
FIG1, axis_ = plt.subplots()
sns.set(color_codes=True)
sns.set_style('whitegrid')
sns.kdeplot(y_prob1, color="g", shade=True, ax=axis_, label="Closure in 1 Year")
sns.kdeplot(y_prob3, color="r", ax=axis_, label="Closure in 3+ Years")
plt.xlim(0,1)
plt.ylabel('Density', fontsize=18)
plt.xlabel('Predicted Probability', fontsize=18)
FIG1.show()
plt.savefig('predicted_probabilities')
plt.close(FIG1)	
    
###################################################33

# Data loading and spliting:
filename = "Data/Processed_Data.csv"
df = pl.open_csv_data(filename)
df = df[df['duration']>=10]
df = df[['tractce10', 'duration']]
df = df.dropna()
avg_duration = df.groupby(['tractce10'], as_index=False).mean()
avg_duration = avg_duration.apply(pd.to_numeric)

#Reading SHP of Mexico City
chicago_link = './Data/boundaries/geo_export_a1dc4961-2107-4939-88d0-817bf020d71c.shp'  
chicago = geopandas.read_file(chicago_link)
chicago['tractce10'] = chicago['tractce10'].astype('int64')
chicago = chicago.merge(avg_duration, how='left', left_on='tractce10', right_on='tractce10')

#chicago.duration = chicago.duration/chicago.duration.max()
#chicago.plot()

#Plotting
fig, ax = plt.subplots(1, figsize = (10, 6))
chicago.plot(column='duration', scheme='quantiles', ax=ax)
plt.savefig('chicago_map')

#########################################################3
fig, ax = plt.subplots(1, figsize = (10, 6))
gdf[(gdf['set']==1)].plot()
chicago.plot(color='black',ax=ax)



#####################################################
### Fig. 6
set = pd.read_csv('Data/jiv.csv')
df2 = pd.read_csv("Data/Processed_Data.csv")
list_r =  ['index', 'sc1', 'sc3', 'Predict Label 1', 'Predict label 3', 'set',
       'index_old', 'ZIP CODE', 'WARD', 'PRECINCT', 'WARD PRECINCT',
       'POLICE DISTRICT', 'LICENSE CODE', 'LICENSE DESCRIPTION',
       'BUSINESS ACTIVITY ID', 'BUSINESS ACTIVITY', 'CONDITIONAL APPROVAL',
       'LICENSE TERM START DATE_x', 'LICENSE STATUS', 'SSA', 'start_year',
       'LICENSE TERM START DATE_y', 'blockce10', 'countyfp10', 'geoid10',
       'name10', 'statefp10', 'tract_bloc', 'tractce10', 'ada1', 'blue1',
       'brn1', 'g1', 'o1', 'p1', 'pexp1', 'pnk1', 'red1', 'ada3', 'blue3',
       'brn3', 'g3', 'o3', 'p3', 'pexp3', 'pnk3', 'red3', 'ada5', 'blue5',
       'brn5', 'g5', 'o5', 'p5', 'pexp5', 'pnk5', 'red5', 'ada10', 'blue10',
       'brn10', 'g10', 'o10', 'p10', 'pexp10', 'pnk10', 'red10', 'less_1_year',
       '1_2_years', '2_3_years', 'more_3_years', 'train_sample',
       'geoid10_bins', 'tractce10_bins', 'ZIP CODE_bins', 'tract_bloc_bins',
       'blockce10_bins', 'LICENSE CODE_bins']
set.columns=list_r
set.set_index('index_old')
df_final =  set.merge(df2, right_index=True, left_index=True)
geometry = [Point(xy) for xy in zip(df_final.LONGITUDE, df_final.LATITUDE)]
df3 = df_final.drop(['LATITUDE', 'LONGITUDE'], axis=1)
crs = {'init': 'epsg:4326'}
gdf = GeoDataFrame(df3, crs=crs, geometry=geometry)
fig, ax = plt.subplots(1, figsize = (10, 6))
chicago.plot(color='white', edgecolor='black',ax=ax)
gdf[(gdf['set']==1)].plot(ax=ax, color='red')