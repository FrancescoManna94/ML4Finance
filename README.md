# ML4Finance: Hybrid BiLSTM-LightGBM Model for Stock Price Prediction
## Overview
This repository contains the implementation of a hybrid predictive model that combines **Bidirectional Long Short-Term Memory (BiLSTM)** networks and **Light Gradient Boosting Machine (LightGBM)** to forecast the daily closing prices of the S&P 500 index. The model leverages the BiLSTM's ability to extract complex temporal features from historical data and technical indicators, feeding them into the LightGBM algorithm for efficient and robust final predictions.   
## Project Objectives
* Accurate Prediction: Forecast the daily closing price of the S&P 500 index using advanced Machine Learning and Deep Learning techniques.
* Feature Extraction: Utilize a BiLSTM network to process sequential data and extract latent temporal features from a 30-day window of market data.
* Robust Forecasting: Employ LightGBM, a highly efficient gradient boosting framework, to handle the extracted features alongside traditional technical indicators, reducing overfitting and improving prediction stability.
* Generalization: Demonstrate the model's ability to generalize across different financial assets, including other indices (e.g., FTSE MIB), individual stocks (e.g., Intesa, Amazon, Ferrari), and commodities (e.g., Gold).
 ## Methodology
* Data Preprocessing & Feature Engineering:
** Dataset: S&P 500 daily data from January 1, 2010, to December 31, 2024.
** Variables: Open, High, Low, Close prices.
** Calculated Technical Indicators: RSI, MACD, ROC, CCI, Stochastic Oscillator, Williams %R, ATR, NATR, TRANGE.
** Normalization: Dynamic scaling over a 30-day rolling window to account for local ranges and seasonality.
* BiLSTM Architecture:Processes 30-day sequential windows with 30 features.   Composed of three bidirectional LSTM layers (256, 64, and 112 units).   Outputs a compressed vector representation of the temporal dynamics.   Hyperparameter tuning performed using Optuna.   LightGBM Regression:Takes as input the latent vector from the BiLSTM and the technical features of the last day in the window.   Configured as a GBDT regressor with optimized hyperparameters (e.g., n_estimators=712, learning_rate=0.0081, max_depth=6).   Predicts the absolute closing price of the following day.   Key Results
The hybrid approach yielded strong predictive performance on the S&P 500 test set (MAE: 0.3643, MSE: 0.2653, RMSE: 0.5151). Notably, the model demonstrated excellent generalization capabilities, performing well across various asset classes with distinct market dynamics, confirming its robustness and potential utility for investors and traders.   Technologies UsedPythonTensorFlow / Keras (BiLSTM implementation)LightGBM (Regression model)Optuna (Hyperparameter optimization)Pandas / NumPy (Data manipulation)
