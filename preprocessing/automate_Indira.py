# automate_Indira.py (versi auto-detect target)

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore


def detect_target(data, target_col: str):
    """Coba deteksi target jika target_col tidak ada di dataset."""
    if target_col in data.columns:
        return target_col
    
    print(f"[WARNING] Kolom target '{target_col}' tidak ditemukan.")
    print(f"Kolom tersedia: {list(data.columns)}")

    # Cari kandidat kolom target: kategorikal / biner
    candidate_targets = []
    for col in data.columns:
        unique_vals = data[col].nunique()
        if unique_vals <= 10:  # biasanya target tidak banyak kelas
            candidate_targets.append((col, unique_vals))
    
    if not candidate_targets:
        raise ValueError("Tidak ditemukan kandidat kolom target otomatis. Harap cek dataset.")
    
    if len(candidate_targets) == 1:
        print(f"[INFO] Otomatis memilih '{candidate_targets[0][0]}' sebagai target (unik: {candidate_targets[0][1]}).")
        return candidate_targets[0][0]
    else:
        print("[INFO] Ditemukan beberapa kandidat target:")
        for col, uniq in candidate_targets:
            print(f"  - {col} (unik: {uniq})")
        raise ValueError("Terlalu banyak kandidat target, harap pilih manual di script.")
    

def preprocess_data(filepath: str, target_col: str, output_dir: str = "preprocessing/namadataset_preprocessing"):
    """
    Melakukan preprocessing otomatis berdasarkan eksperimen sebelumnya.
    """
    # === 1. Load data ===
    data = pd.read_csv(filepath)

    # === 2. Deteksi kolom target ===
    target_col = detect_target(data, target_col)

    # === 3. Drop kolom identifier (jika ada) ===
    if "PatientID" in data.columns:
        data.drop(columns=["PatientID"], inplace=True)

    # === 4. Tangani missing values ===
    data.dropna(inplace=True)

    # === 5. Hapus duplikat ===
    data.drop_duplicates(inplace=True)

    # === 6. Hapus outlier (Z-score) ===
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        z_scores = np.abs(zscore(data[numeric_cols]))
        data = data[(z_scores < 3).all(axis=1)]

    # === 7. Pisahkan fitur & target ===
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # === 8. Encode target jika kategorikal ===
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # === 9. Encode fitur kategorikal ===
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # === 10. Standarisasi fitur numerik ===
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # === 11. Simpan hasil ===
    os.makedirs(output_dir, exist_ok=True)
    processed_data = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
    output_path = os.path.join(output_dir, "dataset_preprocessed.csv")
    processed_data.to_csv(output_path, index=False)

    print(f"[INFO] Hasil preprocessing disimpan ke: {output_path}")
    return X, y


if __name__ == "__main__":
    input_path = "namadataset_raw/dataset.csv"   # dataset mentah
    target_column = "DiseaseStatus"              # target default (jika tidak ada, akan dicari otomatis)
    preprocess_data(input_path, target_column)
