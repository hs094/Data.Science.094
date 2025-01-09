import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time

class DataStream:
    def __init__(self, size=100):
        """Initialize a data stream buffer."""
        self.size = size
        self.data = deque(maxlen=size)
        self.mean = 0
        self.std_dev = 1  # Start with a non-zero std_dev to avoid division by zero
        self.anomalies = []
        self.anomaly_indices = []  # Store indices of detected anomalies

    def add_data(self, value):
        """Add new data point to the stream and update mean and std deviation."""
        self.data.append(value)
        if len(self.data) > 1:
            self.mean = np.mean(self.data)
            self.std_dev = np.std(self.data)

    def detect_anomaly(self, value, index, threshold=3):
        """Detect if the incoming value is an anomaly based on Z-score."""
        if len(self.data) > 1:  # Ensure there's enough data for z-score calculation
            z_score = (value - self.mean) / self.std_dev
            if abs(z_score) > threshold:
                self.anomalies.append(value)
                self.anomaly_indices.append(index)  # Store the index of the anomaly
                return True
        return False

def generate_data_stream(length=1000):
    """Generate simulated data stream with seasonal patterns and noise."""
    base_value = 50
    seasonal_variation = 10
    data_stream = []
    for i in range(length):
        # Simulate seasonal pattern and random noise
        noise = random.uniform(-5, 5)
        value = base_value + seasonal_variation * np.sin(2 * np.pi * i / 100) + noise
        data_stream.append(value)
        # Introduce anomalies randomly
        if random.random() < 0.05:  # 5% chance to introduce an anomaly
            anomaly_value = value + random.uniform(20, 40)  # High anomaly
            data_stream.append(anomaly_value)
    return data_stream

def visualize_stream(data_stream, anomalies, anomaly_indices):
    """Visualize the data stream and detected anomalies."""
    plt.figure(figsize=(10, 5))
    plt.plot(data_stream, label='Data Stream', color='blue')
    if anomalies:
        plt.scatter(anomaly_indices, anomalies, color='red', label='Anomalies', marker='o')
    plt.title('Data Stream Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    # Initialize data stream and generate data
    stream = DataStream(size=100)
    simulated_data = generate_data_stream(1000)

    # Process the data stream
    for index, value in enumerate(simulated_data):
        stream.add_data(value)
        if stream.detect_anomaly(value, index):  # Pass the index of the value
            print(f"Anomaly detected: {value}")

    # Visualize the results
    visualize_stream(simulated_data, stream.anomalies, stream.anomaly_indices)  # Pass anomaly indices

if __name__ == "__main__":
    main()
