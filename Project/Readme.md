# Signal Coverage Maps Using Measurements and Machine Learning

## 📌 Overview
This project explores **signal coverage mapping** by combining:
- **Theoretical Models**: LDPL, Ordinary Kriging (OK), Kriging with Detrending (OKD)  
- **Machine Learning Models**: Linear Regression, Support Vector Regression (SVR), Random Forests  

The goal is to improve wireless coverage prediction by addressing the **limitations of purely theoretical models** using **real-world data and ML techniques**.

---

## 🎯 Problem & Objectives
### Problem
Predicting wireless signal coverage is difficult due to terrain, obstructions, and interference.  
- Theoretical models → Simple, interpretable but limited.  
- Empirical approaches → Rigid and less adaptive.  

### Objectives
1. Implement theoretical signal propagation models.  
2. Collect and preprocess real-world RSSI and GPS data.  
3. Train ML models for prediction.  
4. Compare theoretical vs. ML models (accuracy, efficiency).  
5. Generate and visualize coverage maps.

---

## ⚙️ Methodology
1. **Theoretical Models:** LDPL, OK, OKD in MATLAB.  
2. **Data Collection:** RSSI + GPS + environmental attributes.  
3. **Preprocessing:**  
   - Handle missing values, outliers  
   - Normalize data  
   - Feature engineering (lat/lon, building density, vegetation, time)  
4. **Machine Learning:**  
   - Models: Linear Regression, SVM, Decision Trees, Random Forest  
   - Validation: k-fold cross-validation  
5. **Evaluation Metrics:** RMSE, MAE, R²  
6. **Visualization:** MATLAB heatmaps & geospatial plots  

---

## 📊 Results
- **ML models achieved lower RMSE** than theoretical models.  
- Heatmaps showed better alignment of ML predictions with actual measurements.  
- Visualizations: histograms, scatter plots, 3D plots, and coverage maps.  

---
