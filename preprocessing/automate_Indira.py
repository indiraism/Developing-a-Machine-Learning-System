import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore


def preprocess_data(filepath: str, target_col: str, output_dir: str = "preprocessing/gene_expression_preprocessed.csv"):
    """
    Melakukan preprocessing otomatis berdasarkan eksperimen sebelumnya.

    Parameters
    ----------
    filepath : str
        Path ke dataset CSV.
    target_col : str
        Nama kolom target (label) yang ingin diprediksi.
    output_dir : str
        Folder tempat menyimpan dataset hasil preprocessing.

    Returns
    -------
    X : pd.DataFrame
        Fitur yang sudah diproses (siap latih).
    y : pd.Series
        Label/target yang sudah di-encode (jika kategorikal).
    """

    # === 1. Load data ===
    data = pd.read_csv(filepath)

    # === 2. Drop kolom identifier ===
    if "PatientID" in data.columns:
        data.drop(columns=["PatientID"], inplace=True)

    # === 3. Tangani missing values (drop jika ada NA) ===
    data.dropna(inplace=True)

    # === 4. Hapus duplikat ===
    data.drop_duplicates(inplace=True)

    # === 5. Deteksi & hapus outlier (berdasarkan Z-score pada kolom numerik) ===
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        z_scores = np.abs(zscore(data[numeric_cols]))
        data = data[(z_scores < 3).all(axis=1)]

    # === 6. Pisahkan fitur & target ===
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # === 7. Encoding label target (jika kategorikal) ===
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # === 8. Encoding kolom kategorikal fitur ===
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # === 9. Standardisasi fitur numerik ===
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # === 10. Simpan hasil ke folder output ===
    os.makedirs(output_dir, exist_ok=True)
    processed_data = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
    output_path = os.path.join(output_dir, "dataset_preprocessed.csv")
    processed_data.to_csv(output_path, index=False)

    print(f"Hasil preprocessing disimpan ke: {output_path}")

    return X, y


if __name__ == "__main__":
    # Sesuaikan path dan target sesuai dataset
    input_path = "Gen_Expression_raw/Gene-Expression-Analysis-and-Disease-Relationship.csv"   # dataset mentah
    target_column = "DiseaseStatus"                    # ganti sesuai kolom target
    preprocess_data(input_path, target_column)
