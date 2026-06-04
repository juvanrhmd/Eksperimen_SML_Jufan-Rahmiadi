import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_path):
    print("Memulai otomatisasi preprocessing...")
    
    # 1. Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File mentah tidak ditemukan di: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. Bersihkan Nama Kolom & Duplikat
    df.columns = df.columns.str.strip()
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        
    # 3. Handling Target Mapping
    df['Status'] = df['Status'].map({'Alive': 1, 'Dead': 0})
    
    # 4. Encoding Fitur Kategorikal
    categorical_cols = [
        'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 
        'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status'
    ]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 5. Scaling Fitur Numerik
    numeric_cols = [
        'Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months'
    ]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 6. Simpan Hasil Preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing sukses! Data siap latih disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur folder kriteria 1
    INPUT_FILE = "../Breast_Cancer_Raw.csv"
    OUTPUT_FILE = "./Breast_Cancer_Raw_preprocessing/breast_cancer_clean.csv"
    
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)