# Step 1: Import the required libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Step 2: Create an output folder

os.makedirs("outputs", exist_ok=True)


# Step 3: Load the sales dataset

df = pd.read_csv("sales_data.csv")

print("First five rows of sales data:")
print(df.head())

print("\nDataset information:")
print(df.info())

# Step 4: Clean the dataset

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

df["Sales"] = pd.to_numeric(
    df["Sales"], errors="coerce"
)

df = df.dropna(subset=["Date", "Sales"])

df = df.drop_duplicates()

df = df.sort_values("Date")
df = df.reset_index(drop=True)

if len(df) < 10:
    raise ValueError(
        "Please provide at least 10 valid sales records."
    )

print("\nCleaned sales data:")
print(df.head())

print("\nTotal records:", len(df))

# Step 5: Visualize historical sales

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales Amount")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/sales_trend.png")
plt.show()

# Step 6: Create a numerical feature

df["Day_Number"] = (
    df["Date"] - df["Date"].min()
).dt.days

X = df[["Day_Number"]]
y = df["Sales"]


# Step 7: Split data chronologically

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Step 8: Train the model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed!")


# Step 9: Predict sales for the test data

y_pred = model.predict(X_test)


# Step 10: Calculate evaluation metrics

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# Step 11: Display results

print("\nModel Evaluation")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Step 12: Plot actual vs predicted sales

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"].iloc[split_index:],
    y_test,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    df["Date"].iloc[split_index:],
    y_pred,
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/actual_vs_predicted.png")
plt.show()


# Step 13: Train a final model using all historical data

final_model = LinearRegression()

final_model.fit(X, y)


# Step 14: Create the next 7 dates

future_dates = pd.date_range(
    start=df["Date"].max() + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)


# Step 15: Convert future dates into day numbers

future_day_numbers = (
    future_dates - df["Date"].min()
).days

future_X = pd.DataFrame({
    "Day_Number": future_day_numbers
})


# Step 16: Predict future sales

future_sales = final_model.predict(future_X)


# Step 17: Create the forecast table

forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_sales.round(2)
})

print("\nNext 7 Days Sales Forecast")
print("-------------------------")
print(forecast)


# Step 18: Save forecast results

forecast.to_csv(
    "outputs/future_sales_forecast.csv",
    index=False
)

print("\nForecast saved successfully!")


# Step 19: Visualize future sales

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Sales"],
    label="Historical Sales"
)

plt.plot(
    forecast["Date"],
    forecast["Predicted_Sales"],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

plt.title("Sales Forecasting System")
plt.xlabel("Date")
plt.ylabel("Sales Amount")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/future_sales_graph.png")
plt.show()


print("\nSales Forecasting System completed!")