#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The main script for the analysis of churn predictions
'''
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import pandas as pd
import joblib
# sklean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# custom modules
import config as c
import plotting


def import_data():
    '''
    returns dataframe for the csv found at the data path of the project

    input:
            Empty: The path is created within the function based on the path
            where the projects are located and the project name
    output:
            data_df: pandas dataframe if the data are found, otherwise None.

    TEST function: @test_import
    '''

    path_to_data = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_data()
    # list the csv files in the firectory
    csv_files = list(filter(lambda x: '.csv' in x, os.listdir(path_to_data)))
    # only one file exists in this project
    csv_filename = c.join(path_to_data, csv_files[0])
    # cast to a dataframe
    data_df = pd.read_csv(csv_filename)

    return data_df


def perform_eda(df, path_to_images):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None


    '''

    # Check for missing values
    if df.isnull().sum().any():
        # Define the image path
        path_to_images = c.FetchPaths(
            c.PROJECTS_PATH, c.PROJECT_NAME).to_images()
        # Get the features that contain missing values
        features_with_missing_vals = df.isnull().any()[
            df.isnull().any()].index.tolist()

        c.logging.critical('!!! There are features with missing values')
        c.logging.info(f'Features: {features_with_missing_vals}')
        c.logging.info(f'Types: {df.dtypes[features_with_missing_vals]}')

    else:
        c.logging.info('None of the features has missing values.')

    # Construct the target (or outpout variable) based on the Attrition_Flag
    # column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Drop the feature Attrition Flag, since we renamed it to Churn
    df.drop(columns='Attrition_Flag', inplace=True)

    # plot and save the target distribution
    plotting.plot_target_distribution(df, path_to_images)
    # plot and save the distribution of the age feature (hueded by churn)
    plotting.plot_age_distribution(df, path_to_images)
    # plot the counts of marital status filtered by churn
    plotting.plot_marital_status_counts(df, path_to_images)
    # plot the distribution of feature 'Total_Trans_Ct'
    plotting.plot_Total_Trans_CT_distribution(df, path_to_images)
    # plot the feature correlation matrix
    plotting.plot_feature_correlation(df, path_to_images)

    # return the dataframe that now contains the target column ('Churn')
    return df


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''

    # loop over the corresponding categorical features and replace each value
    # (e.g: "F" in gender) by a numerical value for this feature (e.g: for
    # gender, take the mean churn for F and replace 'F' with that)

    for feature in category_lst:
        # get the available groups (e.g: "M" and "F")
        available_groups = df.groupby(feature).mean()['Churn']
        # encode and create new columns
        df[feature + '_Churn'] = df[feature].map(available_groups.to_dict())
        # drop the old column
        df.drop(columns=feature, inplace=True)

    c.logging.info(f'Categorical features encoded\
                   with frequency encoding: {category_lst}')

    # Select only specific columns to continue processing
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn',
        'Churn']

    df = df[keep_cols]

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # define categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # encode categorical columns=
    df_encoded = encoder_helper(df, cat_columns)

    # Prepare the data for Scikit Learn
    y = df_encoded['Churn']
    X = df_encoded.drop(columns='Churn')

    c.logging.info('Applied simple train-test split with test size 30%')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=c.random_state)

    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test, path_to_images):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    ###############################
    # DEFINE THE CLASSIFIERS
    ###############################
    # random forest
    rfc = RandomForestClassifier(random_state=c.random_state)
    # Logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    ###############################
    # GRID SEARCH
    ###############################
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=c.grids['random_forest'],
        cv=5)

    # Fit the models
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # plot the classification report per classifiers
    plotting.classification_report_image(y_train,
                                         y_test,
                                         y_train_preds_lr,
                                         y_train_preds_rf,
                                         y_test_preds_lr,
                                         y_test_preds_rf,
                                         path_to_images)

    # path to models
    path2models = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_models()

    # store models
    joblib.dump(cv_rfc.best_estimator_,
                os.path.join(path2models, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(path2models,
                                  'logistic_model.pkl'))

    return cv_rfc


# %%
# =============================================================================
# WRAPPER
# =============================================================================
# Define the image path
image_path = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_images()

if __name__ == '__main__':
    # load the data
    DATA_df = import_data()
    # returns the dataframe that now includes the target column
    DATA_df = perform_eda(DATA_df, image_path)
    # prepare data for classification
    X_train, X_test, y_train, y_test = perform_feature_engineering(DATA_df)
    # train models
    model = train_models(X_train, X_test, y_train, y_test, image_path)
    # plot feature importance
    plotting.feature_importance_plot(model, X_test, image_path)
