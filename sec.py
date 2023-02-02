import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the salary data from the CSV file
salary_data = pd.read_csv("Salary_Data.csv")

# Divide the data into training and testing sets
X = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Train a linear regression model using the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the mean squared error between the actual and predicted values
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the training data as a scatter plot, with the regression line overlaid
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plot the test data as a scatter plot, with the regression line from the training data overlaid
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()