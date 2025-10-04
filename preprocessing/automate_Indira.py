import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

def preprocess_data(input_path, target_col):
    # === 1. Load data ===
    print(f"📂 Membaca dataset dari: {input_path}")
    data = pd.read_csv(input_path)

    print("🔍 Kolom yang tersedia:", data.columns.tolist())

    # === 2. Cek apakah target_col ada di dataset ===
    if target_col not in data.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan di dataset!")

    # === 3. Tangani missing values ===
    data = data.dropna()

    # === 4. Pisahkan fitur (X) dan target (y) ===
    y = data[target_col]
    X = data.drop(columns=[target_col])

    # === 5. Standarisasi hanya untuk kolom numerik di X ===
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # === 6. Gabungkan kembali X dan y ===
    processed = pd.concat([X, y], axis=1)

    # === 7. Simpan hasil ===
    output_path = os.path.join("preprocessing", "namadataset_preprocessing", "data_processed.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed.to_csv(output_path, index=False)

    print(f"✅ Preprocessing selesai. Dataset tersimpan di: {output_path}")


if __name__ == "__main__":
    # Pastikan ada 2 argumen: path dataset dan nama kolom target
    if len(sys.argv) != 3:
        print("Usage: python preprocessing/automate_Indira.py <input_path> <target_column>")
        sys.exit(1)

    input_path = sys.argv[1]
    target_column = sys.argv[2]

    preprocess_data(input_path, target_column)
