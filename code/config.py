#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module that includes:
    1. generic, re-usable functions/classes (i.e: class to create dirs)
    2. hyperparameters (i.e: the random seed)
    3. The logging configuration


Created on Thu May 12 18:04:32 2022

@author: Christos
"""
# =============================================================================
# MODULES & ALLIASES
# =============================================================================
import os
import logging


# alliases
join = os.path.join


class FetchPaths():
    '''
    Simple class to get the paths. All paths are returned as str type.
    Attributes:
        1. projects_path
        2. project name
    '''

    def __init__(self, projects_path, project_name,):
        self.projects_path = projects_path
        self.project_name = project_name

    def to_project(self):
        '''
        Returns the project path.
        '''
        return join(self.projects_path, self.project_name,)

    def to_data(self):
        '''
        Returns the path where the data are stored.
        '''
        return join(self.projects_path, self.project_name, 'data')

    def to_images(self):
        '''
        Returns the path where the images are stored as .png files.
        '''
        return join(self.projects_path, self.project_name, 'images')

    def to_logs(self):
        '''
        Returns the path where the logs are stored as .log files.
        '''
        return join(self.projects_path, self.project_name, 'logs')

    def to_models(self):
        '''
        Returns the path where the models are stored as .pkl files.
        '''
        return join(self.projects_path, self.project_name, 'models')

    def __str__(self):
        return f'Project: {self.project_name}'


# =============================================================================
# PROJECT ATTRIBUTES
# =============================================================================
# The PROJECTS_PATH is where the code for all running projects are stored
PROJECTS_PATH = '/Users/christoszacharopoulos/projects/'
PROJECT_NAME = 'customer_churn_prediction'


# =============================================================================
# SET UP THE LOGGING CONFIGURATION
# =============================================================================
log_filename=join(FetchPaths(PROJECTS_PATH,PROJECT_NAME).to_logs(),
                  'results.log')

# set up the logging file
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')

# unicode characters to log success and errors
error = 4 * '\u274C'
success = 4 * '\u2705'


# =============================================================================
# HYPERPARAMETERS 
# =============================================================================
random_state = 42

# add grid parameters per classifier
grids={}
# Random Forest
grids['random_forest'] = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}
# Logist Regression
grids['logreg']={
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,800,100)),
    #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}











