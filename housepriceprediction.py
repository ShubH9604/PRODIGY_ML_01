import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set up seaborn for better graphs
sns.set(style="whitegrid")

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
square_footage = np.random.randint(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)

# Create a price based on a formula with some noise
price = (square_footage * 300) + (bedrooms * 50000) + (bathrooms * 30000) + np.random.randint(-50000, 50000, n_samples)

# Create a DataFrame
df = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': price
})

# Format the Price column as dollars in the DataFrame
df['Price'] = df['Price'].apply(lambda x: f'${x:,.0f}')

# Display the first 50 rows of the dataset
df.head(50)

# Scatter plot of Price vs Square Footage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SquareFootage', y=df['Price'].str.replace('$', '').str.replace(',', '').astype(float), data=df)

# Format the y-axis with dollar sign, commas, and zero decimal places
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.title('Price vs Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

# Histogram of the number of bedrooms
plt.figure(figsize=(10, 6))
sns.histplot(df['Bedrooms'], bins=20, kde=False)

# Format the y-axis with commas and zero decimal places
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.title('Distribution of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.show()

# Convert 'Price' column back to numeric for plotting
df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '').astype(float)

# Pairplot of all features
sns.pairplot(df)
plt.show()

# Features (X) and Target (y)
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Scatter plot of Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Format the axes with dollar sign and commas
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Function to predict house price based on input features
def predict_price(square_footage, bedrooms, bathrooms):
    prediction = model.predict([[square_footage, bedrooms, bathrooms]])
    return prediction[0]

# Example Prediction
example_square_footage = 2500
example_bedrooms = 4
example_bathrooms = 3

predicted_price = predict_price(example_square_footage, example_bedrooms, example_bathrooms)
print(f"The predicted price for a house with {example_square_footage} sqft, {example_bedrooms} bedrooms, and {example_bathrooms} bathrooms is: ${predicted_price:,.0f}")

# Get user input
square_footage = float(input("Enter the square footage of the house: "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))

# Predict the price
predicted_price = predict_price(square_footage, bedrooms, bathrooms)

# Display the predicted price
print(f"The predicted price of the house is: ${predicted_price:,.0f}")

import pickle

# Save the trained model to a file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to house_price_model.pkl")

import pickle

# Load the model from the file
with open('house_price_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
print("Model loaded from house_price_model.pkl")

# Test the loaded model with an example prediction
predicted_price = loaded_model.predict([[2500, 4, 3]])
print(f"The predicted price for a house with 2500 sqft, 4 bedrooms, and 3 bathrooms is: ${predicted_price[0]:,.0f}")

