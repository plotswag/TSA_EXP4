# Ex.No:04 FIT ARMA MODEL FOR TIME SERIES

## Developed by: JEEVANESH S
## Register Number: 212222243002
## Date: 16-09-2025


**AIM:**  
To implement ARMA model in python.  

**ALGORITHM:**  
1. Import necessary libraries.  
2. Set up matplotlib settings for figure size.  
3. Define an ARMA(1,1) process with coefficients a=1 and ma=1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.  
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.  
5. Define an ARMA(2,2) process with coefficients a=2 and ma=2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.  
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.  

**PROGRAM:** 
```
# Import necessary Modules and Functions  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statsmodels.tsa.arima.model import ARIMA  
from statsmodels.tsa.arima_process import ArmaProcess  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  

# Load the dataset, perform data exploration
data = pd.read_csv('/content/cardekho.csv', parse_dates=['sale_date'], index_col='sale_date')

print("Dataset Head:")
print(data.head())

# Resample to monthly frequency
data_monthly = data.resample('MS').sum()

# Declare required variables and set figure size, and visualise the data  
N = 1000  
plt.rcParams['figure.figsize'] = [12, 6]

X = data_monthly
plt.plot(X)
plt.title('Original Car Sales Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

print(f"ARMA(1,1) Parameters - AR: {phi1_arma11:.4f}, MA: {theta1_arma11:.4f}")

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

print(f"ARMA(2,2) Parameters - AR1: {phi1_arma22:.4f}, AR2: {phi2_arma22:.4f}, MA1: {theta1_arma22:.4f}, MA2: {theta2_arma22:.4f}")

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
## OUTPUT:
<img width="917" height="495" alt="image" src="https://github.com/user-attachments/assets/3ac5a4d1-974e-4566-b194-8095b5bd4314" />
<img width="1125" height="575" alt="image" src="https://github.com/user-attachments/assets/d61a0936-374b-4096-a32b-9da93c516b23" />
<img width="928" height="492" alt="image" src="https://github.com/user-attachments/assets/68377a58-102b-482f-9337-df14c48a9b68" />
<img width="935" height="535" alt="image" src="https://github.com/user-attachments/assets/af3c9a72-086b-4279-8604-16b6872a3be7" />
<img width="945" height="512" alt="image" src="https://github.com/user-attachments/assets/9acf23a4-2950-4602-b505-ec021df9c1f0" />
<img width="932" height="500" alt="image" src="https://github.com/user-attachments/assets/6dc192d6-5f6d-4971-be71-0c7e9d1f3496" />
<img width="932" height="525" alt="image" src="https://github.com/user-attachments/assets/ee8ba798-1d46-4890-8912-1a77a7ca5a76" />
<img width="946" height="507" alt="image" src="https://github.com/user-attachments/assets/e8b90578-b301-45c4-adbd-77bd7c0db80d" />


## RESULT:
Thus, a python program is created to fit ARMA Model successfully for car sales data.
