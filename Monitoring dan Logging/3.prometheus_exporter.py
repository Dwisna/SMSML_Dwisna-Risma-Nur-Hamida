from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
with open('../Membangun_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_path = '../Membangun_model/mlruns/1/models/m-39393ff580834d80bd7dca3265a8d93d/artifacts/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Definisi metrik
REQUEST_COUNT = Counter('ml_request_count_total', 'Total request ke model')
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Latency request model')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Akurasi model')
PREDICTION_COUNT = Counter('ml_prediction_count_total', 'Total prediksi', ['result'])

# Set akurasi model
MODEL_ACCURACY.set(0.90)

def predict(gender, age, salary):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        data = pd.DataFrame([[gender, age, salary]], 
                           columns=['Gender', 'Age', 'EstimatedSalary'])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
    
    result = 'beli' if prediction == 1 else 'tidak_beli'
    PREDICTION_COUNT.labels(result=result).inc()
    return result

if __name__ == '__main__':
    start_http_server(8001)
    print("✅ Prometheus exporter berjalan di port 8001")
    print("Buka http://localhost:8001/metrics")
    
    # Test data
    test_data = [
        (1, 35, 80000),
        (0, 25, 30000),
        (1, 45, 120000),
        (0, 30, 50000),
        (1, 50, 150000),
    ]
    
    i = 0
    while True:
        gender, age, salary = test_data[i % len(test_data)]
        result = predict(gender, age, salary)
        print(f"Prediksi: {result}")
        i += 1
        time.sleep(1)