# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# Sample data: Area in square feet (X) and corresponding house price in $1000s (y)
# For simplicity, let's simulate a linear relationship: price = 50 * area + some noise
X = np.array([[650], [800], [950], [1100], [1250], [1400], [1550], [1700], [1850], [2000]])
y = np.array([325, 400, 475, 550, 625, 700, 775, 850, 925, 1000])  # Prices in $1000s
 
# Create a linear regression model
model = LinearRegression()
 
# Train the model with the data
model.fit(X, y)
 
# Make predictions on the same data to visualize the line of best fit
y_pred = model.predict(X)
 
# Print model parameters
print("Model Coefficient (Slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
 
# Predict price for a new house (e.g., 1600 sq. ft)
new_area = [[1600]]
predicted_price = model.predict(new_area)[0]
print(f"\nPredicted price for 1600 sq.ft: ${predicted_price * 1000:.2f}")
 
# Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.scatter(new_area, predicted_price, color='green', label='Prediction (1600 sq.ft)')
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price ($1000s)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Figure_1.png')
