import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score #Need for the benchmark evaluation
import matplotlib.pyplot as plt

# DATA:
ticker = 'ZUO'

# Load the Google stock data
data = yf.download(ticker, start='2018-05-05', end='2024-04-05')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Define a function to make future predictions
def make_future_predictions(model, X_test, steps_to_predict):
    # Get the last input sequence from the test data
    last_input = X_test[-1].reshape(1, X_test.shape[1], 1)

    # Initialize the predicted prices list
    predicted_prices = []

    # Make predictions for the specified number of steps
    for _ in range(steps_to_predict):
        # Make a prediction using the last input sequence
        next_price = model.predict(last_input)[0, 0]

        # Append the predicted price to the list
        predicted_prices.append(next_price)

        # Update the last input sequence with the new prediction
        last_input = np.concatenate((last_input[:, 1:, :], np.expand_dims([[next_price]], axis=1)), axis=1)
        last_input = last_input.reshape(1, X_test.shape[1], 1)

    # Rescale the predicted prices back to the original scale
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    return predicted_prices

# Prepare the data for LSTM
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 30 #This cannot be greather than len(test_data) <-- IMPORTANT!!

X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Reshape the input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the input shape
input_shape = (X_train.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(Input(shape=input_shape))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', loss)

# Make predictions
y_pred = model.predict(X_test)

# Rescale the predictions
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print the results
#print('Actual Closing Prices:', y_test)
#print('Predicted Closing Prices:', y_pred.flatten())

#This is for a benchmark evaluation against a simple baseline model
y_baseline = y_test[:-1]
baseline_mse = mean_squared_error(y_test[1:], y_baseline)
baseline_r2 = r2_score(y_test[1:], y_baseline)
print('Baseline MSE:', baseline_mse)
print('Baseline R-squared:', baseline_r2)

# LSTM model evaluation
lstm_mse = mean_squared_error(y_test, y_pred)
lstm_r2 = r2_score(y_test, y_pred)
print('LSTM MSE:', lstm_mse)
print('LSTM R-squared:', lstm_r2)

# printing the Deltas
mse_delta = round(baseline_mse - lstm_mse,2)
if mse_delta < 0: mse_delta=f"\033[1;31m{mse_delta}\033[0m"
print(f'MSE delta: {mse_delta}')
R_delta = round(lstm_r2 - baseline_r2,2)
if R_delta < 0: R_delta=f"\033[1;31m{R_delta}\033[0m"
print(f'R^2 delta: {R_delta}')

# Plot the actual and predicted closing prices
tck = yf.Ticker(ticker)
company_name = tck.info['longName']
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Closing Prices')
plt.plot(y_pred.flatten(), label='Predicted Closing Prices')
plt.title(f'{company_name} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
#plt.show()

# Make future predictions
steps_to_predict = 7
future_prices = make_future_predictions(model, X_test, steps_to_predict)

# Plot the future predictions
plt.plot(range(len(y_test), len(y_test) + steps_to_predict), future_prices.flatten(), label='Future Predicted Prices')
plt.legend()
plt.show()

#For a good result in favor of LSTM: Baseline-MSE > LSTM-MSE and Baseline-Rsq < LSTM-Rsq
#example
#Baseline MSE: 0.5
#Baseline R-squared: 0.8
#LSTM MSE: 0.3
#LSTM R-squared: 0.9
if baseline_mse > lstm_mse and baseline_r2 < lstm_r2:
    print("LSTM is performing better at predicting!!")
else:
    print("LSTM not good")