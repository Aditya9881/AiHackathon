import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

def generate_property_data(num_entries=100):
    cities = ["Pune", "Mumbai", "Lucknow", "Delhi", "Bangalore", "Hyderabad"]
    data = []
    for _ in range(num_entries):
        location = random.choice(cities)
        area_sqft = random.randint(500, 3000)
        num_bedrooms = random.randint(1, 5)
        num_bathrooms = random.randint(1, 4)
        year_built = random.randint(2015, 2024)
        price = area_sqft * random.randint(2000, 10000)
        
        # Randomly assign distances 
        near_metro = random.uniform(1,5)    
        near_railway = random.uniform(1,5) 
        near_mall = random.uniform(2,10) 
        near_medical = random.uniform(0.5,3)
        
        data.append({
            "location": location,
            "area_sqft": area_sqft,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "year_built": year_built,
            "price": price,
            "near_metro": near_metro,
            "near_railway": near_railway,
            "near_mall": near_mall,
            "near_medical": near_medical
        })
    return pd.DataFrame(data)

# Generate the dataset
data = generate_property_data(100)  
print("Dataset Generated Dynamically!")
print(data.head())


data = data.fillna(method='ffill')
features = ['location', 'area_sqft', 'num_bedrooms', 'num_bathrooms', 'year_built', 'near_metro', 'near_railway', 'near_mall', 'near_medical']
target = 'price'


data = pd.get_dummies(data, columns=['location'], drop_first=True)
X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

print("\nDataset Insights:")

location_names = [col for col in data.columns if col.startswith("location_")]
average_prices = {}
for location in location_names:
    average_prices[location.replace('location_', '')] = data[data[location] == 1]['price'].mean()

print("\nAverage Price by Location:")
print(average_prices)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted Property Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()

feature_importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.title("Feature Importance in Property Price Prediction")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, bins=30, color='purple')
plt.title("Distribution of Property Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

print("\nProvide the following details to predict property price:")

area_sqft = float(input("Enter the area in square feet (500 - 3000): "))
num_bedrooms = int(input("Enter the number of bedrooms (1-5): "))
num_bathrooms = int(input("Enter the number of bathrooms (1-4): "))

min_year = data['year_built'].min()
max_year = data['year_built'].max()
while True:
    year_built = int(input(f"Enter the year the property was built (between {min_year} and {max_year}): "))
    if min_year <= year_built <= max_year:
        break
    print(f"Invalid year! Please enter a year between {min_year} and {max_year}.")

unique_locations = [col for col in data.columns if col.startswith("location_")]
print("\nAvailable Locations:")
for i, loc in enumerate(unique_locations, start=1):
    print(f"{i}. {loc.replace('location_', '')}")

location_choice = int(input("Select the location (Enter the number): "))
selected_location = unique_locations[location_choice - 1]

print("\nSummary of Inputs:")
print(f"Area (sq ft): {area_sqft}")
print(f"Bedrooms: {num_bedrooms}")
print(f"Bathrooms: {num_bathrooms}")
print(f"Year Built: {year_built}")
print(f"Location: {selected_location.replace('location_', '')}")

near_metro = float(input("Enter the distance to the nearest metro station (in km, e.g., 1.0): "))
near_railway = float(input("Enter the distance to the nearest railway station (in km, e.g., 2.0): "))
near_mall = float(input("Enter the distance to the nearest mall (in km, e.g., 3.0): "))
near_medical = float(input("Enter the distance to the nearest medical shop (in km, e.g., 0.5): "))

new_data = {col: 0 for col in X.columns}
new_data['area_sqft'] = area_sqft
new_data['num_bedrooms'] = num_bedrooms
new_data['num_bathrooms'] = num_bathrooms
new_data['year_built'] = year_built
new_data[selected_location] = 1  
new_data['near_metro'] = near_metro
new_data['near_railway'] = near_railway
new_data['near_mall'] = near_mall
new_data['near_medical'] = near_medical

new_data_df = pd.DataFrame([new_data])

predicted_price = model.predict(new_data_df)
confidence_interval = [predicted_price[0] * 0.95, predicted_price[0] * 1.05]
print(f"\nThe predicted property price is: ₹{predicted_price[0]:,.2f}")
print(f"Price Confidence Interval: ₹{confidence_interval[0]:,.2f} - ₹{confidence_interval[1]:,.2f}")

plt.figure(figsize=(6, 4))
plt.bar(['Predicted Price'], [predicted_price[0]], color='green')
plt.title("Predicted Property Price")
plt.ylabel("Price")
plt.show()

similar_properties = data.loc[
    (data['area_sqft'] >= area_sqft - 200) & (data['area_sqft'] <= area_sqft + 200) & 
    (data['num_bedrooms'] >= num_bedrooms - 1) & (data['num_bedrooms'] <= num_bedrooms + 1) & 
    (data['num_bathrooms'] >= num_bathrooms - 1) & (data['num_bathrooms'] <= num_bathrooms + 1)
].head(3)

print("\nTop 3 Similar Properties from the Dataset:")
print(similar_properties[['area_sqft', 'num_bedrooms', 'num_bathrooms', 'year_built', 'price']])
