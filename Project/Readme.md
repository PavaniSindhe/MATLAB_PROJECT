SIGNAL COVERAGE MAPS USING MEASUREMENTS AND MACHINE LEARNING 

📑 Project Overview

Focus: Generating signal coverage maps using theoretical models (LDPL, Ordinary Kriging, OKD) and machine learning models (Linear Regression, SVR, Random Forest).

Goal: Compare prediction accuracy, interpretability, and adaptability of both approaches for wireless networks.

🎯 Problem & Objectives

Problem:

Predicting wireless signal coverage is challenging due to environmental factors like terrain, obstructions, and interference.

Pure theoretical models oversimplify; traditional empirical models lack adaptability.

Objectives:

Implement theoretical propagation models.

Collect real-world measurement data (RSSI, GPS, environment).

Train machine learning models.

Compare accuracy and computational efficiency.

Generate signal coverage maps using both approaches.

⚙️ Methodology

Theoretical Models: LDPL, OK, OKD implemented in MATLAB.

Data Collection: Real-world RSSI measurements with GPS coordinates.

Preprocessing:

Handle missing values & outliers.

Normalize numerical/categorical features.

Feature engineering (lat, lon, building density, vegetation, time).

Machine Learning:

Models: Linear Regression, SVM, Decision Trees, Random Forest.

Validation: k-fold cross-validation.

Evaluation Metrics: RMSE, MAE, R².

Visualization: MATLAB Mapping Toolbox for coverage heatmaps.

📊 Results

Histograms, scatter plots, and boxplots illustrate RSSI distribution.

Heatmaps compare LDPL, OK, OKD coverage maps.

Machine learning showed lower RMSE than theoretical models.

Final maps: ML predictions were closer to actual measurements.
