#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the unit tests for the main function
churn_library
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import time
import pandas as pd
import config as c
import churn_main as cls


def test_import():
    """
        Check that the data can be  found and loaded
        """
    path_to_data = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_data()
    # list the csv files in the firectory
    csv_files = list(filter(lambda x: '.csv' in x, os.listdir(path_to_data)))

    try:
        # check that the data folder contains csv files
        assert len(csv_files) != 0
        csv_filename = c.join(path_to_data, csv_files[0])
        # load the files into the workspace
        start_time = time.time()

        data_df = pd.read_csv(csv_filename)
        end_time = time.time()
        try:
            # check that the .csv files are not empty
            assert len(data_df.index) != 0
            c.logging.info(f'{c.success} Data was successfully loaded.')
            c.logging.info(f'Loading time: {end_time-start_time:1.2f}s')
            c.logging.info(
                f'Size: {data_df.shape[0]} rows X {data_df.shape[1]} columns')
            c.logging.info(f'File loaded: {csv_filename}')

            # return the DATA to the workspace
            return data_df
        except AssertionError:
            c.logging.error(
                f'{c.error}: Could not load dataset into the workspace')
            # stop the script if the data were not found
            raise SystemExit(
                f'{c.error} Could not load data. Operation terminated. \
                    See {c.log_filename} for details')

    except AssertionError:
        c.logging.error(
            f'{c.error}: Could not find any .csv files in {path_to_data}')
        # stop the script if the data were not found
        raise SystemExit(
            f'{c.error} Could not load data. Operation terminated. \
                See {c.log_filename} for details')


def test_eda():
    '''
    test perform eda function
        '''
    # Define the image path
    path_to_images = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_images()
    try:
        assert os.path.exists(path_to_images)
        c.logging.info(f'Image path exists @: {path_to_images}')
    except AssertionError:
        c.logging.error(
            f'{c.error} - Image path does not exist - {path_to_images}')
        raise SystemExit(f'{path_to_images} Does not exist')
    # load the data
    df = cls.import_data()
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError:
        c.logging.error('The input to the EDA function is not a DataFrame')
        raise SystemExit('perform_eda requires a DataFrame as input')


def test_encoder_helper():
    '''
        test encoder helper
        '''
    # define categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        
        'Card_Category'
    ]
    # load the data
    df = cls.import_data()
    # check if cat_columns exist in the given dataframe
    try:
        assert set(cat_columns) <= set(df.columns)
    except AssertionError:
        c.logging.error('The specified categorical columns are not part \
                        of the provided dataframe')


def test_perform_feature_engineering():
    '''
        test perform_feature_engineering
        '''

    # load the data
    df = cls.import_data()

    # Prepare the data for Scikit Learn
    y = df['Attrition_Flag']
    X = df.drop(columns='Attrition_Flag')

    try:
        assert y.shape[0] == X.shape[0]
    except AssertionError:
        c.logging.error('Error in X and y dims. Probably some NaN files\
                        are automatically droped')


def test_train_models():
    '''
        test train_models
        '''
    # Define the models path
    path2models = c.FetchPaths(c.PROJECTS_PATH, c.PROJECT_NAME).to_models()
    try:
        assert os.path.exists(path2models)
        c.logging.info(f'Model path exists @: {path2models}')
    except AssertionError:
        c.logging.error(
            f'{c.error} - Model path does not exist - {path2models}')
        raise SystemExit(f'{path2models} Does not exist')

    try:
        pd.read_pickle(os.path.join(path2models,
                                    os.listdir(path2models)[0]))
        pd.read_pickle((os.path.join(path2models,
                                     os.listdir(path2models)[1])))
    except AssertionError:
        c.logging.error(f'Models not found at: {path2models}')
