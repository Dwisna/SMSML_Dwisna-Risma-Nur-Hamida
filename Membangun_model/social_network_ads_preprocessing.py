import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')
print("✅ Dataset berhasil dimuat!")
print("Shape:", df.shape)

# Hapus kolom User ID
df = df.drop(columns=['User ID'])

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
print("✅ Encoding Gender selesai!")

# Pisahkan fitur dan target
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data train: {X_train.shape[0]} baris")
print(f"Data test : {X_test.shape[0]} baris")

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Preprocessing selesai!")
print("✅ Scaler disimpan sebagai scaler.pkl")