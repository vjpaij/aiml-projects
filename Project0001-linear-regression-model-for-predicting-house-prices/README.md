### Description:

A Linear Regression model predicts continuous values—like house prices—based on input features such as size, number of bedrooms, location 
etc. This basic model uses a synthetic dataset to demonstrate how to predict house prices based on the house's area in square feet.

This basic linear regression model learns a relationship between house size and price. For real-world datasets, consider using the Boston Housing Dataset, Kaggle housing data, or incorporate features like location, condition, and age of property.

# House Price Prediction using Simple Linear Regression

This Python script demonstrates a simple linear regression model to predict house prices based on their area in square feet. It uses the `scikit-learn` library for the machine learning model and `matplotlib` for data visualization.

## How it Works

The core idea is to find a linear relationship between the input feature (house area) and the target variable (house price). The model learns this relationship from the sample data and can then be used to predict the price for new, unseen data points.

The formula for a simple linear regression line is:

y\=mx+c  

*   `y`: The predicted value (house price).
    
*   `m`: The **coefficient** or **slope** of the line. It represents the change in price for a one-unit change in area.
    
*   `x`: The input feature (area of the house).
    
*   `c`: The **intercept**. It's the value of `y` when `x` is 0.
    

## Code Breakdown

### 1\. Importing Libraries

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    

*   **`numpy`**: Used for efficient numerical operations, especially creating the arrays for our data.
    
*   **`matplotlib.pyplot`**: A comprehensive library for creating static, animated, and interactive visualizations in Python. We use it to plot our data points and the regression line.
    
*   **`sklearn.linear_model.LinearRegression`**: This class contains the algorithm for performing Linear Regression.
    
*   **`sklearn.metrics.mean_squared_error`**: A function to calculate the Mean Squared Error (MSE), which is a common metric to evaluate the performance of a regression model.
    

### 2\. Sample Data

    X = np.array([[650], [800], [950], [1100], [1250], [1400], [1550], [1700], [1850], [2000]])
    y = np.array([325, 400, 475, 550, 625, 700, 775, 850, 925, 1000])
    

*   **`X` (Feature)**: A 2D NumPy array representing the area of houses in square feet. `scikit-learn` models expect the features (`X`) to be in a 2D array format.
    
*   **`y` (Target)**: A 1D NumPy array representing the corresponding house prices in thousands of dollars.
    

### 3\. Model Training

    # Create a linear regression model
    model = LinearRegression()
     
    # Train the model with the data
    model.fit(X, y)
    

*   An instance of the `LinearRegression` model is created.
    
*   The `model.fit(X, y)` method trains the model. It takes the feature data (`X`) and target data (`y`) and calculates the optimal values for the coefficient (`m`) and intercept (`c`) that best fit the data.
    

### 4\. Making Predictions & Evaluating the Model

    # Make predictions on the same data to visualize the line of best fit
    y_pred = model.predict(X)
     
    # Print model parameters
    print("Model Coefficient (Slope):", model.coef_[0])
    print("Model Intercept:", model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y, y_pred))
    

*   `model.predict(X)` uses the trained model to predict the house prices for the input areas `X`. The result `y_pred` represents the points that lie on the calculated regression line.
    
*   **`model.coef_`**: This attribute stores the slope(s) of the regression line. For simple linear regression, it contains a single value.
    
*   **`model.intercept_`**: This attribute stores the intercept of the regression line.
    
*   **`mean_squared_error(y, y_pred)`**: This function calculates the MSE between the actual prices (`y`) and the prices predicted by the model (`y_pred`). MSE is the average of the squared differences between the actual and predicted values. A lower MSE indicates a better fit.
    

### 5\. Predicting a New Value

    # Predict price for a new house (e.g., 1600 sq. ft)
    new_area = [[1600]]
    predicted_price = model.predict(new_area)[0]
    print(f"\nPredicted price for 1600 sq.ft: ${predicted_price * 1000:.2f}")
    

*   This section demonstrates the practical use of the trained model.
    
*   We create a new data point, `new_area`, for a house of 1600 sq. ft. Note that it's in a 2D array format `[[1600]]`.
    
*   The `model.predict()` method is called on this new data to get the predicted price.
    
*   The result is then formatted into a user-friendly string.
    

### 6\. Visualization

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
    

*   **`plt.scatter(X, y, ...)`**: Creates a scatter plot of the original data points (blue dots).
    
*   **`plt.plot(X, y_pred, ...)`**: Draws the line of best fit (red line) that the model has learned.
    
*   **`plt.scatter(new_area, ...)`**: Plots the specific prediction for the 1600 sq. ft. house as a distinct point (green dot).
    
*   The rest of the `plt` commands add labels, a title, a legend, and a grid to make the plot readable.
    
*   **`plt.show()`**: Displays the final plot.