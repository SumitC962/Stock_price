import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load and preprocess dataset
data = pd.read_csv('stock_price.csv') 

# Handle missing values by forward filling
data.fillna(method='ffill', inplace=True)

# Sort data by date
data = data.sort_values('Date')

# Create additional features
data['Lag1'] = data['Close'].shift(1)  # Previous day's close
data['Lag2'] = data['Close'].shift(2)  # Close two days ago
data['Change'] = data['Close'].pct_change()  # Daily percentage change
data.dropna(inplace=True)  

# Select features and target (Close price)
features = data[['Open', 'High', 'Low', 'Volume', 'Lag1', 'Lag2', 'Change']]
target = data['Close']

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# --------- Linear Regression Model ---------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# --------- Random Forest Model with Hyperparameter Tuning ---------
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf_model = rf_random.best_estimator_
rf_predictions = best_rf_model.predict(X_test)

# --------- LSTM Model ---------
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, return_sequences=True, input_shape=(1, X_train.shape[1])))
lstm_model.add(LSTM(units=64, return_sequences=False))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model with history
history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Make predictions using the LSTM model
lstm_predictions = lstm_model.predict(X_test_lstm)

# Reverse Scaling of Predictions to Original Scale
linear_predictions_rescaled = linear_predictions
rf_predictions_rescaled = rf_predictions
lstm_predictions_rescaled = lstm_predictions.flatten()  # Flatten for easier plotting

# --------- Model Evaluation ---------
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R² (Accuracy): {r2 * 100:.2f}%')
    return mae, rmse, r2

# Evaluate all three models with R² Accuracy
evaluate_model(y_test, linear_predictions, 'Linear Regression')
evaluate_model(y_test, rf_predictions, 'Random Forest')
evaluate_model(y_test, lstm_predictions_rescaled, 'LSTM')

# --------- Visualization of Predictions ---------
plt.figure(figsize=(14, 7))

# Plot actual stock prices
plt.plot(y_test.values, label='Actual Prices', color='blue')

# Plot predictions from the Linear Regression model
plt.plot(linear_predictions, label='Linear Regression Predictions', linestyle='dashed', color='green')

# Plot predictions from the LSTM model
plt.plot(lstm_predictions_rescaled, label='LSTM Predictions', linestyle='dashed', color='red')

# Plot predictions from the Random Forest model
plt.plot(rf_predictions, label='Random Forest Predictions', linestyle='dashed', color='orange')

# Customize the plot
plt.title('Predicted vs Actual Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# --------- Plot Training Loss for LSTM ---------
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --------- Save the Models ---------
joblib.dump(linear_model, 'linear_model.pkl')
joblib.dump(best_rf_model, 'rf_model.pkl')
lstm_model.save('lstm_model.h5')

print("Models saved successfully!")

