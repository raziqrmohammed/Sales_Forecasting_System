Sales Forecasting System
Python script that analyzes historical sales, evaluates a linear regression model, and forecasts sales for the next seven days.
Project files
- sales_forecasting.py — loads and cleans the data, trains the model, evaluates it, and creates charts and a forecast.
- sales_data.csv — sales records with Date and Sales columns.
- outputs/ — created automatically when the script runs; contains charts and the forecast CSV.
Requirements
Python 3 and these packages:
pip install pandas numpy matplotlib scikit-learn
Run
From the project folder, run:
python sales_forecasting.py
The script expects sales_data.csv in the current working directory. It requires at least 10 valid sales records.
What it does
1. Reads and cleans the sales data.
2. Plots the historical sales trend.
3. Trains a linear regression model using the first 80% of records, in date order.
4. Evaluates predictions on the remaining 20% using MAE, RMSE, and R².
5. Trains a final model on all records and forecasts the next seven calendar days.
Output files
After running, the outputs/ folder contains:
- sales_trend.png — historical sales.
- actual_vs_predicted.png — actual and test-period predicted sales.
- future_sales_forecast.csv — forecast dates and predicted sales.
- future_sales_graph.png — historical sales with the seven-day forecast.
Data format
The CSV must include these columns:
Date,Sales
2025-01-01,1000
2025-01-02,1050
Dates should be parseable as dates, and sales should be numeric. Invalid rows and duplicate rows are removed automatically.
Notes
The model uses elapsed calendar days as its only input feature, so it assumes sales follow a roughly linear trend over time. It does not account for seasonality, holidays, promotions, or other factors that can affect sales.


7:41 PM
