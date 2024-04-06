# Importing necessary libraries
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
import os

# Ignoring warnings
warnings.filterwarnings('ignore')

# Setting display format for float values
pd.options.display.float_format = '${:,.2f}'.format

# Getting today's date and start date for data retrieval
today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'

# Downloading Ethereum price data from Yahoo Finance
eth_df = yf.download('ETH-USD', start_date, today)

# Displaying the last few rows of the dataframe
eth_df.tail()

# Displaying information about the dataframe
eth_df.info()

# Checking for missing values in the dataframe
eth_df.isnull().sum()

# Getting the column names of the dataframe
eth_df.columns

# Resetting the index of the dataframe
eth_df.reset_index(inplace=True)

# Getting the column names of the dataframe after resetting the index
eth_df.columns 

# Creating a new dataframe with only the "Date" and "Open" columns
df = eth_df[["Date", "Open"]]

# Renaming the columns of the dataframe
new_names = {
    "Date": "ds", 
    "Open": "y",
}
df.rename(columns=new_names, inplace=True)

# Displaying the last few rows of the modified dataframe
df.tail()

# ================================ Time Series Plot for Ethereum Open Prices ================================

# Extracting the x and y values for the plot
x = df["ds"]
y = df["y"]

# Creating a new plot figure
fig = go.Figure()

# Adding a scatter trace to the plot
fig.add_trace(go.Scatter(x=x, y=y))

# Setting the title of the plot
fig.update_layout(
    title_text="Time series plot of Ethereum Open Price",
)

# Adding a range selector and rangeslider to the x-axis of the plot
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)

# ================================ Prophet Model ================================

# Creating a Prophet model with multiplicative seasonality
m = Prophet(
    seasonality_mode="multiplicative" 
)

# Fitting the model to the data
m.fit(df)

# Creating a dataframe for future predictions
future = m.make_future_dataframe(periods=365)
future.tail()

# Generating predictions using the trained model
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Predicting the price for the next day
next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
forecast[forecast['ds'] == next_day]['yhat'].item()

# Plotting the forecasted values
plot_plotly(m, forecast)

# Plotting the components of the forecasted values
plot_components_plotly(m, forecast)
