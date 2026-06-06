import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Load Dataset ===
df = pd.read_csv('Social_Network_Ads.csv')
print("✅ Dataset berhasil dimuat!")
print("Shape:", df.shape)

# === Preprocessing ===
df = df.drop(columns=['User ID'])
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# === MLflow Tracking ===
mlflow.set_experiment("Social_Network_Ads_Classification")

# Autolog - otomatis merekam semua parameter dan metrik
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model berhasil dilatih!")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))