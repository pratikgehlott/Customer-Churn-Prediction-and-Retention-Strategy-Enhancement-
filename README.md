# Customer Churn Prediction
"Predict Customer Churn: Leveraging W&B, MLflow, and Hydra for Enhanced Model Performance and Experiment Tracking" is a machine learning project focused on predicting customer attrition, helping businesses identify at-risk customers and implement effective retention strategies. This project emphasizes the importance of experiment tracking, reproducibility, and hyperparameter tuning in achieving optimal model performance.

The project utilizes Weights & Biases (W&B) for real-time monitoring of training metrics, enabling model performance visualization and comparison across different iterations. MLflow is employed for comprehensive experiment tracking, versioning, and model management, while Hydra simplifies configuration and hyperparameter tuning, streamlining the experimentation process.

By leveraging these advanced tools, the project showcases best practices in customer churn prediction and model optimization. Developers and data scientists can learn how to effectively build, evaluate, and fine-tune their models while maintaining organization and transparency throughout the experimentation process. This ultimately results in improved model performance and better insights into customer behavior, empowering businesses to take targeted action to retain valuable customers.

## Project Description
Module to determine which credit card users are most likely to leave. A Python package for a machine learning project that adheres to coding standards (PEP8) and engineering best practices for software implementation is part of the finished project (modular, documented, and tested). Additionally, the package can be run interactively or through a command-line interface (CLI).

## Files and data description
<p align="center">
  <img src="/images/dir_tree.png" width="550" title="Project structure">
</p>

## Running Files
  1. Install all dependencies
  ```
  python -m pip install -r requirements_py3.8.txt
  ```
  2. Run the main analysis script
  ```
  python churn_main.py
  ```
  Two additional modules are called:
    
    * config.py --> The configuration file creates the paths and holds all globals recquired for the analysis. 
    * plotting.py --> All plotting functions are stored in this module


