'''All necessary packages'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


'''
For a better comparison, I have used two algorithms for detecting anomalies from a continuos stream of real time data.
First off, I have used the Long Short-Term Memory Model as it handles sequential data effectively. Before working on the 
real-time data, I trained my model on 10000 data points that I generated and after successful training of the model, I began 
with testing my model on the real data. LSTM is good at learning patterns in a data and predicting the next value in a sequence.
----------------------------------------------------------------------------------------------------------------------------------
Next, I used Z-Score statistical analysis for identifying the outliers from the data stream. I used a sliding window to calculate 
the mean and standard deviation of the recent data points. And as I have declared a threshold, so if a value exceeds that predefined
threshold then it is marked as an anomaly. It's relatively easier to implement as compared to the LSTM. And LSTM requires computational 
resources for training the data as well. 
'''


# LSTM model for anomaly detection
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # Number of features in the hidden state
        self.num_layers = num_layers  # Number of stacked LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        # Fully connected layer to get the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # Forward propagate LSTM
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# For detecting anomalies using LSTM model


class LSTMAnomalyDetector:
    def __init__(self, window_size=100, threshold=5.0):
        self.window_size = window_size  # Size of the sliding window for time series data
        self.threshold = threshold  # Threshold for anomaly detection
        # To store the time series data in a sliding window
        self.data_window = deque(maxlen=window_size)
        self.model = LSTM().to(torch.device('cpu'))
        self.optimizer = optim.Adam(
            self.model.parameters())  # Using Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss

    def fit(self, data):
        self.model.train()  # Set the model to training mode
        # Create sequences for training the LSTM
        sequences = np.array([data[i:i+self.window_size]
                             for i in range(len(data) - self.window_size)])
        x_train = torch.FloatTensor(
            sequences[:, :-1]).view(-1, self.window_size-1, 1).to(torch.device('cpu'))
        y_train = torch.FloatTensor(
            sequences[:, -1]).view(-1, 1).to(torch.device('cpu'))

        # Specifying number of epochs for training
        for epoch in range(5):
            epoch_loss = 0
            for x, y in zip(x_train, y_train):
                x = x.view(1, -1, 1)
                y = y.view(1, 1)
                self.optimizer.zero_grad()  # Clear gradients
                output = self.model(x)  # Forward pass
                loss = self.criterion(output, y)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/5, Loss: {epoch_loss / len(x_train)}")

    def update(self, value):
        # Update the sliding window with new value
        self.data_window.append(value)

    def is_anomaly(self, value):
        if len(self.data_window) < self.window_size:
            return False

        data_array = np.array(list(self.data_window))
        x = torch.FloatTensor(data_array).view(
            1, -1, 1).to(torch.device('cpu'))
        with torch.no_grad():
            prediction = self.model(x).item()  # Predict the next value

        error = abs(value - prediction)
        return error > self.threshold  # Mark as an anomaly if error exceeds threshold

# For detecting anomalies using statistical method - zscore


class ZScore:
    def __init__(self, window_size=100, threshold=3.0):
        self.window_size = window_size  # Size of the sliding window for time series data
        self.threshold = threshold  # Z-score threshold for anomaly detection
        # To store the time series data in a sliding window
        self.data_window = deque(maxlen=window_size)

    def update(self, value):
        # Update the sliding window with new value
        self.data_window.append(value)

    def is_anomaly(self, value):
        if len(self.data_window) < self.window_size:
            return False  # Not enough data to detect anomaly

        mean = np.mean(self.data_window)
        std = np.std(self.data_window)
        z_score = abs((value - mean) / std)  # Calculate the z-score
        return z_score > self.threshold  # Anomaly if z-score exceeds threshold

# Function to generate heart rate data


def generate_data_stream(n_points, anomalies=True):
    t = np.arange(n_points)
    base_rate = 70

    # Variation based on daily activities
    daily_variation = 5 * np.sin(2 * np.pi * (t % 1440) / 1440)

    # Weekly variation
    weekly_variation = 3 * np.sin(2 * np.pi * (t % (1440*7)) / (1440*7))

    # Adding variations to the base rate
    heart_rate = base_rate + daily_variation + weekly_variation

    # Adding random noise to the data
    heart_rate += np.random.normal(0, 2, n_points)

    # Adding anomalies to the data
    if anomalies:
        anomaly_indices = np.random.choice(
            n_points, size=int(0.01 * n_points), replace=False)
        heart_rate[anomaly_indices] += np.random.choice([-1, 1], size=len(
            anomaly_indices)) * np.random.uniform(15, 30, size=len(anomaly_indices))

    # Limit the values between 40 and 120 bpm
    heart_rate = np.clip(heart_rate, 40, 120)

    return heart_rate


def main():
    print("Generating data for training LSTM model...")
    # Generating 10000 datapoints for training the LSTM model
    train_data = generate_data_stream(10000, anomalies=False)

    print("Training LSTM model...")
    window_size = 100

    lstm_detector = LSTMAnomalyDetector(window_size=window_size, threshold=5.0)
    lstm_detector.fit(train_data)

    zscore_detector = ZScore(window_size=window_size, threshold=3.0)

    print("Model training completed. Starting real-time detection...")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))
    fig.subplots_adjust(hspace=0.4)  # Add space between plots
    ax1.set_title("LSTM Heart Rate Anomaly Detection")
    ax2.set_title("Z-Score Heart Rate Anomaly Detection")
    for ax in (ax1, ax2):
        ax.set_xlabel("Time")
        ax.set_ylabel("Heart Rate (bpm)")

    line1, = ax1.plot([], [], label='Heart Rate')
    anomaly_scatter1, = ax1.plot([], [], 'ro', label='Anomalies')
    line2, = ax2.plot([], [], label='Heart Rate')
    anomaly_scatter2, = ax2.plot([], [], 'ro', label='Anomalies')

    for ax in (ax1, ax2):
        ax.legend()
        ax.set_xlim(0, 200)
        ax.set_ylim(40, 120)

    data = []
    anomalies_lstm = []
    anomaly_times_lstm = []
    anomalies_zscore = []
    anomaly_times_zscore = []

    try:
        # Real-time detection for 1000 data points
        for i in range(1000):
            value = generate_data_stream(1, anomalies=True)[0]
            data.append(value)

            is_anomaly_lstm = lstm_detector.is_anomaly(value)
            is_anomaly_zscore = zscore_detector.is_anomaly(value)

            lstm_detector.update(value)
            zscore_detector.update(value)

            if is_anomaly_lstm:
                anomalies_lstm.append(value)
                anomaly_times_lstm.append(i)

            if is_anomaly_zscore:
                anomalies_zscore.append(value)
                anomaly_times_zscore.append(i)

            # Updating plots
            line1.set_data(range(len(data)), data)
            anomaly_scatter1.set_data(anomaly_times_lstm, anomalies_lstm)
            line2.set_data(range(len(data)), data)
            anomaly_scatter2.set_data(anomaly_times_zscore, anomalies_zscore)

            if i > 200:
                ax1.set_xlim(i-200, i)
                ax2.set_xlim(i-200, i)
            else:
                ax1.set_xlim(0, 200)
                ax2.set_xlim(0, 200)

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopped by user")

    # For disabling the interactive mode
    plt.ioff()
    plt.show()


# Main script
if __name__ == "__main__":
    main()
