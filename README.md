# **Stock Price Prediction: ARIMA, ARIMAX + ANN, and LSTM Models**

## **Project Overview**

This project explores stock price prediction for **IBM's daily closing prices** using time series and machine learning models. We aim to address both linear and nonlinear components of financial data, incorporating key market indicators such as **interest rates**, the **S&P 500 index**, and **treasury yields**.

The models developed include:
1. **ARIMA**: Benchmark linear time-series model.
2. **ARIMAX + ANN**: A hybrid model combining ARIMAX predictions with ANN (Artificial Neural Network) for residual learning.
3. **LSTM**: A deep learning model designed to capture long-term dependencies and nonlinear patterns in time-series data.

---

## **Motivation**

Stock price prediction is complex due to the chaotic and nonlinear nature of financial markets. By combining statistical models with machine learning, we aim to:
- Better capture **trends** and **volatility**.
- Leverage exogenous features (e.g., interest rates, S&P 500) for improved predictions.
- Explore hybrid approaches that combine **ARIMAX** and **ANN** for residual learning.

---
## **Contribution**
- Marie Qi: ARIMAX and ANN model
- Han Gao: LSTM, ARIMAX + ANN, and data preprocessing
- Jiaming Liu: ARIMA, ARIMAX Validation, ARIAM + ANN
---
## **Acknowledgements**
This project is part of the NYU Advance topic: Deep Learning course. Special thanks to the course TA and professor for their guidance.

---
## **Data Collection and Preprocessing**

- **Dataset**: IBM daily closing stock prices (1999–2024).
- **Source**: Alpha Vantage API and historical records.
- **Additional Features**:
  - **Interest rates**
  - **S&P 500 index**
  - **Treasury yields**
- **Preprocessing**:
  - Missing values were filled using **linear interpolation**.
  - **MinMaxScaler** applied for ANN and LSTM models.
  - Tested stationarity using **ADF** and **KPSS** tests.
  - Differecing was applied to handle non-stationarity

---

## **Models**

### **1. ARIMA (Benchmark Model)**

- **Purpose**: Establish a linear baseline for comparison.
- **Approach**:
  - Used **AutoARIMA** to select optimal parameters (p, d, q).
  - Captured linear trends using ARIMA(2,1,0).
- **Results**:
  - **RMSE**: 7.56
  - **MAPE**: 3.41%
  - **Explained Variance**: 68.2%

---

### **2. ARIMAX + ANN (Hybrid Model)**

- **Purpose**: Combine ARIMAX with ANN to handle nonlinear residuals.
- **Method**:
  - **ARIMAX**: Used S&P 500, interest rates, and treasury yields as exogenous variables.
  - **ANN**: Feedforward neural network (2 hidden layers) trained on ARIMAX residuals.
- **Key Technologies**: Auto_Arima for ARIMAX, Hyperband for ANN hyperparameter tuning.
- **Results of ARIMAX**:
  - **RMSE**: 16.570
  - **MAPE**: 11.321%
  - **Explained Variance**: 88.7%
- **Results of ARIMAX+ANN**:
  - **RMSE**: 16.541
  - **MAPE**: 11.314%
  - **Explained Variance**: 88.9%
  
---

### **3. LSTM (Deep Learning Model)**

- **Purpose**: Use LSTM to compare with above models
- **Method**:
  - Used sliding window sequences with a **look-back window of 30 days**.
  - LSTM architecture: **3 layers** with 256, 128, and 64 units.
  - Applied **Exponential Moving Averages (EMA)** for data augmentation.
- **Results**:
  - **RMSE**: 6.686
  - **MAPE**: 4.052%
  - **Explained Variance**:94.595%

---

## **Key Findings**

- **ARIMAX + ANN**: ARIMAX performs well on the dataset, but ANN struggles to improve upon ARIMAX due to lack of clear seasonal trends in the dataset
- **LSTM**: LSTM performs well in both long term and short term predictions and is the best model among all. 

---

## **Future Work**

1. **Hybrid Models**: Explore ARIMA + LSTM for better performance.
2. **Feature Engineering**: Add more exogenous features (e.g., GDP, dividends).
3. **Data**: Collect larger datasets to improve deep learning model generalization.
4. **Validation**: Address potential data leakage and overfitting and ensure rigorous cross-validation.

