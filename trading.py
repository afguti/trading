import yfinance as yf
import mplfinance as mpf
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# DATA:
ticker = input("Ticker: ").upper()
mode = input("Swim Trade mode by default. Enter Y for Day Trade [Y]: ").upper()
interval = "1m"
entry = False
if mode == 'Y':
    start_date = input("Start date [YYYYMMDDHHMM]: ")
    end_date = input("End Time [HHMM]: ")
    interval = input("Interval by default 1m [2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]: ")
    print(f"Python3 ./trading.py {start_date} d {end_date} {ticker} {interval}")
    end_date = datetime(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]), int(end_date[:2]), int(end_date[2:]))
    start_date1 = datetime(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]), int(start_date[8:10]), int(start_date[10:]))
else:    
    start_date = input("Start date [YYYYMMDD]: ") #Has to be one day less than end_date
    #end_date = input("")'2024-04-05' #Change HERE to run the backtest
    take_profit = 8.63
    entry = True

# Download stock data
data = yf.download(ticker, start=start_date1, end=end_date, interval=interval)

# Function to find support and resistance levels
def find_levels(data, threshold=0.005):
    # Find local maxima
    max_peaks, _ = find_peaks(data['High'], distance=5, prominence=threshold * np.max(data['High']))
    # Find local minima
    min_peaks, _ = find_peaks(-data['Low'], distance=5, prominence=threshold * np.max(data['Low']))
    
    resistance_levels = data['High'].iloc[max_peaks]
    support_levels = data['Low'].iloc[min_peaks]
    
    return resistance_levels, support_levels

# Function to calculate stop loss and take profit levels based on risk/reward ratio
def calc_risk_reward_levels(entry_price, take_profit, risk_reward_ratio=2):
    # Set the delta risk
    risk = abs(entry_price-take_profit)/risk_reward_ratio
    # Here we define the stop_loss wheter it is short or long
    if entry_price > take_profit:
        stop_loss = entry_price+risk
    else:
        stop_loss = entry_price-risk

    return stop_loss

# Example entry price. We are going to use the last Close price recorded
entry_price = data['Close'].iloc[-1]  # Replace with the actual entry price

# Calculate stop loss and take profit levels based on the risk/reward ratio. The last varible is to define short or long
if entry == True: stop_loss = calc_risk_reward_levels(entry_price, take_profit, risk_reward_ratio=2)

# Find support and resistance levels
resistance_levels, support_levels = find_levels(data)

# Addind SMA indicators
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()
# Calculate the 20-day Exponential Moving Average (EMA)
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
# Calculate the 50-day Exponential Moving Average (EMA)
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

# Create additional plots for support, resistance, entry, stop loss, and take profit
addplot = []

# Support levels with thinner lines
for level in support_levels:
    addplot.append(mpf.make_addplot([level] * len(data), color='green', linestyle='--', alpha=0.5, width=1))

# Resistance levels with thinner lines
for level in resistance_levels:
    addplot.append(mpf.make_addplot([level] * len(data), color='red', linestyle='--', alpha=0.5, width=1))

# Adding the SMA20
addplot.append(mpf.make_addplot(
        data['SMA20'],
        color='orange',
        linestyle='-',
        width=1,
        label='20-Day SMA'))
# Adding the EMA50
addplot.append(mpf.make_addplot(
        data['EMA50'],
        color='pink',
        linestyle='-',
        width=1,
        label='50-Day EMA'))

addplot.append(mpf.make_addplot([entry_price] * len(data), color='blue', linestyle='--', width=1, label=f'Close price: {entry_price:.2f}'))

if entry == True:
    # Entry, stop loss, and take profit with defined risk/reward ratio
    addplot.append(mpf.make_addplot([entry_price] * len(data), color='blue', linestyle='-', width=1, label=f'Entry Price {entry_price:.2f}'))
    addplot.append(mpf.make_addplot([stop_loss] * len(data), color='red', linestyle='-', width=1, label=f'Stop Loss {stop_loss:.2f}'))
    addplot.append(mpf.make_addplot([take_profit] * len(data), color='green', linestyle='-', width=1, label=f'Take Profit {take_profit:.2f}'))
    # Shading specification
    fill_between = [
        {
            "y1": entry_price,
            "y2": take_profit,
            "color": "green",
            "alpha": 0.3,
        },
        {
            "y1": entry_price,
            "y2": stop_loss,
            "color": "red",
            "alpha": 0.3,
        }
    ]
else:
    fill_between = [
        {
            "y1": entry_price,
            "y2": entry_price,
        },
        {
            "y1": entry_price,
            "y2": entry_price,
        }
    ]

# Plot the candlestick chart with support/resistance, entry price, stop loss, and take profit
tck = yf.Ticker(ticker)
company_name = tck.info['longName']
mpf.plot(data, type='candle', volume=True, style='charles', 
         title=f"[{ticker}] "+company_name,
         ylabel='Price (USD)', ylabel_lower='Volume',
         addplot=addplot,fill_between=fill_between,figratio=(10, 6),figscale=1.2)