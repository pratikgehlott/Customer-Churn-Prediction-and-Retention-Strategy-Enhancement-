#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 00:28:02 2022

@author: Christos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.metrics import classification_report
import os


def plot_target_distribution(df, pth):

    fig = plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)
    df['Churn'].hist()
    plt.xlim([0, 1])
    plt.ylabel('# customers')
    plt.xlabel('customer churn')
    plt.xticks([0, 1])
    plt.title(
        'Distribution of target values',
        style='oblique',
        fontweight='bold')
    sns.despine(trim=False, offset=10)
    plt.grid(False)
    fig.tight_layout(pad=2)

    plt.savefig(
        fname=os.path.join(
            pth,
            'target_distribution.png'),
        bbox_inches='tight')
    plt.show()


def plot_age_distribution(df, pth):

    fig = plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)
    sns.displot(data=df, x='Customer_Age', hue='Churn',
                hue_order=[1, 0], palette={0: 'blue', 1: 'red'})
    plt.ylabel('# customers')
    plt.xlabel('age')
    plt.title(
        'Distribution of customer age',
        style='oblique',
        fontweight='bold')
    sns.despine(trim=False, offset=10)
    plt.grid(False)
    fig.tight_layout(pad=2)

    plt.savefig(
        fname=os.path.join(
            pth,
            'age_distribution.png'),
        bbox_inches='tight')
    plt.show()


def plot_marital_status_counts(df, pth):

    fig = plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)
    sns.histplot(data=df, x='Marital_Status', multiple="dodge", hue='Churn',
                 stat='density', shrink=0.8, common_norm=False)
    plt.ylabel('Normalized counts')
    plt.xlabel('Marital Status')
    plt.title(
        'Customer marital status filtered by churn',
        style='oblique',
        fontweight='bold')
    sns.despine(trim=False, offset=10)
    plt.grid(False)

    plt.savefig(
        fname=os.path.join(
            pth,
            'marital_status.png'),
        bbox_inches='tight')
    plt.show()


def plot_Total_Trans_CT_distribution(df, pth):

    fig = plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)

    sns.histplot(data=df, x='Total_Trans_Ct',
                 hue='Churn', stat='density', kde=True,
                 hue_order=[1, 0], palette={0: 'blue', 1: 'red'})
    plt.ylabel('# customers')
    plt.xlabel('Total_Trans_Ct')
    sns.despine(trim=False, offset=10)
    plt.grid(False)
    fig.tight_layout(pad=2)
    plt.savefig(
        fname=os.path.join(
            pth,
            'total_trans_ct.png'),
        bbox_inches='tight')
    plt.show()


def plot_feature_correlation(df, pth):

    fig = plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)

    plt.subplot(121)
    mask = np.triu(np.ones_like(df[df.Churn == 0].corr()))
    sns.heatmap(df[df.Churn == 0].corr(), annot=False, cmap='RdBu_r',
                linewidths=2, cbar=False, mask=mask)
    plt.title('Churn=0', style='oblique', fontweight='bold')
    plt.subplot(122)
    sns.heatmap(df[df.Churn == 1].corr(), annot=False, cmap='RdBu_r',
                linewidths=2, mask=mask)
    plt.title('Churn=1', style='oblique', fontweight='bold')
    plt.suptitle('Feature correlation')
    plt.savefig(
        fname=os.path.join(
            pth,
            'feature_correlation.png'),
        bbox_inches='tight')
    plt.show()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                pth):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Random forest
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        fname=os.path.join(
            pth,
            'classification_report_random_forest.png'),
        bbox_inches='tight')
    plt.show()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        fname=os.path.join(
            pth,
            'classification_report_logistic_regression.png'),
        bbox_inches='tight')
    plt.show()


def feature_importance_plot(model, X_test, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # fig 1
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # fig 2
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), names, rotation=90)

    plt.savefig(
        fname=os.path.join(
            output_pth,
            'feature_importance.png'),
        bbox_inches='tight')
    plt.show()
