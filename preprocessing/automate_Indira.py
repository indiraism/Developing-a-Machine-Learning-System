import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_path):
    # Baca dataset
    print(f"📂 Membaca dataset dari: {input_path}")
    if input_path.endswith(".csv"):
        data = pd.read_csv(input_path)
    elif input_path.endswith(".xlsx"):
        data = pd.read_excel(input_path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan .csv atau .xlsx")

    print("✅ Dataset berhasil dibaca.")
    print(f"🔎 Jumlah baris: {data.shape[0]}, Kolom: {data.shape[1]}")

    # Cek missing values
    if data.isnull().sum().sum() > 0:
        print("⚠️ Ada missing values, akan diisi dengan median.")
        data = data.fillna(data.median(numeric_only=True))

    # Scaling fitur numerik
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        print(f"⚡ Normalisasi fitur numerik selesai: {list(numeric_cols)}")

    # Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".csv"):
        data.to_csv(output_path, index=False)
    else:
        data.to_excel(output_path, index=False)

    print(f"💾 Dataset hasil preprocessing disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated preprocessing script")
    parser.add_argument("--dataset", type=str, required=True, help="Path ke file dataset input")
    parser.add_argument("--output", type=str, default="preprocessing/output_preprocessed.xlsx",
                        help="Path untuk menyimpan hasil preprocessing")
    args = parser.parse_args()

    preprocess_data(args.dataset, args.output)
