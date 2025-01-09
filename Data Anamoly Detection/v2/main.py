import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from sklearn.ensemble import IsolationForest
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataStream:
    def __init__(self, size=100):
        """Initialize a data stream buffer."""
        self.size = size 
        self.data = deque(maxlen=size)
        self.anomalies = []
        self.anomaly_indices = []
        self.model = IsolationForest(contamination=0.05)  # Set contamination to 5%
        
    def add_data(self, value, index):
        """Add new data point to the stream and update the model."""
        self.data.append(value)
        if len(self.data) >= self.size:
            self.detect_anomalies(index)  # Pass the index for anomaly detection

    def detect_anomalies(self, current_index):
        """Detect anomalies using Isolation Forest."""
        if len(self.data) < self.size:
            return
        
        # Convert deque to DataFrame for model input
        data_array = np.array(self.data).reshape(-1, 1)
        predictions = self.model.fit_predict(data_array)
        
        # Check for anomalies
        for index, prediction in enumerate(predictions):
            if prediction == -1:  # Anomaly detected
                anomaly_value = self.data[index]
                anomaly_index = current_index - self.size + index  # Correctly calculate global index
                self.anomalies.append(anomaly_value)
                self.anomaly_indices.append(anomaly_index)
                logging.info(f"Anomaly detected: {anomaly_value}")

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
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label='Data Stream', color='blue', alpha=0.5)
    if anomalies:
        plt.scatter(anomaly_indices, anomalies, color='red', label='Anomalies', marker='o')
    plt.title('Data Stream Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.axhline(y=np.mean(data_stream), color='green', linestyle='--', label='Mean Value')  # Mean line
    plt.legend()
    plt.grid()
    plt.savefig('file2.png')
    plt.show()

def main():
    # Initialize data stream and generate data
    stream = DataStream(size=100)
    simulated_data = generate_data_stream(1000)

    # Process the data stream
    for index, value in enumerate(simulated_data):
        stream.add_data(value, index)  # Pass the current index to add_data

    # Visualize the results
    visualize_stream(simulated_data, stream.anomalies, stream.anomaly_indices)

if __name__ == "__main__":
    main()
