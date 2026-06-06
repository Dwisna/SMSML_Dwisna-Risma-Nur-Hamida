from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random

# Definisi metrik
REQUEST_COUNT = Counter('ml_request_count', 'Total request ke model')
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Latency request model')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Akurasi model')
PREDICTION_COUNT = Counter('ml_prediction_count', 'Total prediksi', ['result'])

def simulate_request():
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        time.sleep(random.uniform(0.01, 0.1))
    result = random.choice(['beli', 'tidak_beli'])
    PREDICTION_COUNT.labels(result=result).inc()
    MODEL_ACCURACY.set(0.90)

if __name__ == '__main__':
    start_http_server(8001)
    print("✅ Prometheus exporter berjalan di port 8001")
    print("Buka http://localhost:8001/metrics")
    while True:
        simulate_request()
        time.sleep(1)