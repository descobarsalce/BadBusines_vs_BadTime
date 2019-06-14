# BadBusines_vs_BadTime
## ML for PP - Final Project

Code to implement business survival analysis for the city of Chicago.

## Execution Files:
There are three workfile:
### 1 - 06_07_business_licenses.py
This script is main file to build the dataset using the 06_12_get_data.py and run the different models using the auxiliary files.

### 2 - 06_12_get_data.py
This script create the data set using a Class and generate the database in which the models are going to run.

### 3 - predict_values.py
This script do the following:
1 - Creates the output used for the cross analysis between the three year prediction and one year prediciton.
2 - Creates the different maps used in the project

## Auxiliaxy Files:

This code loops over several machine learning classifiers implemented using sklearn package. The model also features data pre-processing semi-automatic (i.e. after choosing certain parameters and classifying a few specific variables types, such as outcomes). It also test several hyperparameters for each model and selects the best one based on the preferred metric.

There are three grid sizes:

test: to test if things are working small: if you've got less than an hour large: if you've got time or cores

The package is modular so more models or hyperparameters can be easily added.

There are three files in the folder:

pipeline.py - This file includes several pre-processing functions to clean the data, split into test-test, create quantiles, create dummy variables, etc.

pipeline_models.py - This includes the code for fitting the models and going through different sets of hyperparameters. It also relies on pipeline.py to do some data pre-processing if called directly, although the functions can also be used one by one.

Models implemented: all the models from sklearn are supported. Metrics implemented: precision at k%, recall at k%, f1 at k%, area under ROC curve

Models currently implemented: ['RF', 'DT', 'LR']

## Folders

There are different folders in the repository:

### Data
Wiew README file in data folder

### Report
Contain the resports of project.

### Results
Contain the results of the scripts and the different maps and images created of the final report
