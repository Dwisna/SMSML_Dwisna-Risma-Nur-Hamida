import pandas as pd
import pickle

# Load scaler
with open('../Membangun_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load model
model_path = '../Membangun_model/mlruns/1/models/m-39393ff580834d80bd7dca3265a8d93d/artifacts/model.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Contoh data untuk prediksi
data = {
    'Gender': [1],
    'Age': [35],
    'EstimatedSalary': [80000]
}

df = pd.DataFrame(data)
df_scaled = scaler.transform(df)

prediction = model.predict(df_scaled)
result = "Beli ✅" if prediction[0] == 1 else "Tidak Beli ❌"

print("=== Hasil Prediksi ===")
print(f"Gender: {'Male' if data['Gender'][0] == 1 else 'Female'}")
print(f"Age: {data['Age'][0]}")
print(f"Estimated Salary: {data['EstimatedSalary'][0]}")
print(f"Prediksi: {result}")