import yfinance as yf
import mplfinance as mpf
import numpy as np
from scipy.signal import find_peaks

# DATA:
ticker = 'ZUO'
start_date = '2024-04-04' #Has to be one day less than end_date
end_date = '2024-04-05' #Change HERE to run the backtest
take_profit = 8.63
entry = True

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

# Download stock data
data = yf.download(ticker, start='2024-01-05', end=end_date)

# Example entry price. We are going to use the last Close price recorded
entry_price = data.loc[start_date, 'Close']  # Replace with the actual entry price

# Calculate stop loss and take profit levels based on the risk/reward ratio. The last varible is to define short or long
stop_loss = calc_risk_reward_levels(entry_price, take_profit, risk_reward_ratio=2)

# Find support and resistance levels
resistance_levels, support_levels = find_levels(data)

# Addind SMA indicators
data['SMA20'] = data['Close'].rolling(window=20).mean()
# Calculate the 20-day Exponential Moving Average (EMA)
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

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
# Adding the EMA20
addplot.append(mpf.make_addplot(
        data['EMA20'],
        color='pink',
        linestyle='-',
        width=1,
        label='20-Day EMA'))

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
