### Description:

ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model used for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) to predict future values based on historical data. In this project, we use statsmodels to build and evaluate an ARIMA model on a sample time series dataset.

- Simulates a real-world trend with seasonal noise
- Builds an ARIMA(1,1,1) model for forecasting
- Compares predicted vs. actual values and evaluates using MSE

## Time Series Forecasting with ARIMA - README

This project demonstrates a basic implementation of time series forecasting using the ARIMA model. It involves generating synthetic data, fitting an ARIMA model, and visualizing the forecast.

### Prerequisites

Ensure the following Python libraries are installed:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

### Code Explanation

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```

These libraries are used for data manipulation (`numpy`, `pandas`), visualization (`matplotlib`), time series modeling (`statsmodels`), and model evaluation (`sklearn`).

```python
# Generate synthetic time series data
np.random.seed(42)
months = pd.date_range(start="2022-01", periods=36, freq="M")
sales = np.linspace(200, 300, 36) + np.random.normal(scale=10, size=36)
df = pd.DataFrame({"Date": months, "Sales": sales}).set_index("Date")
```

We generate a synthetic monthly sales dataset with a linear upward trend and added Gaussian noise to simulate real-world fluctuations.

```python
# Plot original sales data
df.plot(title="Monthly Sales Over Time", figsize=(10, 4))
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```

A line plot is created to visualize the synthetic sales data over time.

```python
# Split into train and test sets
train = df.iloc[:-6]  # First 30 months
test = df.iloc[-6:]   # Last 6 months
```

The data is split into a training set (used for modeling) and a test set (used for validation).

```python
# Fit ARIMA model
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
```

We create and fit an ARIMA(1,1,1) model:

* `p=1`: Auto-Regressive term
* `d=1`: First-order differencing
* `q=1`: Moving Average term

```python
# Forecast future values
forecast = model_fit.forecast(steps=6)
forecast = pd.Series(forecast, index=test.index)
```

The model forecasts the next 6 time steps, matching the test set length.

```python
# Plot forecast vs actual
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train')
plt.plot(test, label='Test', color='orange')
plt.plot(forecast, label='Forecast', color='green', linestyle='--')
plt.title("ARIMA Forecast vs Actual")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

We visualize how well the ARIMA model forecast matches the actual values.

```python
# Evaluate the forecast
print("\nForecasted Values:\n", forecast)
print("\nMean Squared Error (MSE):", mean_squared_error(test, forecast))
```

The forecast values are printed, and the model performance is evaluated using Mean Squared Error (MSE).

### Summary

This example highlights a simple pipeline for:

* Creating synthetic time series data
* Modeling with ARIMA
* Forecasting future values
* Visualizing results
* Evaluating model performance

It serves as a foundational example for time series forecasting tasks.
