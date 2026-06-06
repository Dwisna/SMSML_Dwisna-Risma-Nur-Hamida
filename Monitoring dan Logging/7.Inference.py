import pandas as pd
import pickle
import mlflow.sklearn

# Load model
model = mlflow.sklearn.load_model(
    "../Membangun_model/mlruns/1/artifacts/random_forest_model"
)

# Load scaler
with open('../Membangun_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Contoh data untuk prediksi
data = {
    'Gender': [1],        # 1 = Male, 0 = Female
    'Age': [35],
    'EstimatedSalary': [80000]
}

df = pd.DataFrame(data)

# Scaling
df_scaled = scaler.transform(df)

# Prediksi
prediction = model.predict(df_scaled)
result = "Beli ✅" if prediction[0] == 1 else "Tidak Beli ❌"

print("=== Hasil Prediksi ===")
print(f"Gender: {'Male' if data['Gender'][0] == 1 else 'Female'}")
print(f"Age: {data['Age'][0]}")
print(f"Estimated Salary: {data['EstimatedSalary'][0]}")
print(f"Prediksi: {result}")