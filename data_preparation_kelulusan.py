# PERTEMUAN 4
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # Langkah 1: Buat dataset CSV (untuk demo kita buat langsung dari dict)
    data = {
        'IPK': [3.8, 2.5, 3.4, 2.1, 3.9, 2.8, 3.2, 2.7, 3.6, 2.3],
        'Jumlah_Absensi': [3,8,4,12,2,6,5,7,4,9],
        'Waktu_Belajar_Jam': [10,5,7,2,12,4,8,3,9,4],
        'Lulus': [1,0,1,0,1,0,1,0,1,0]
    }
    df = pd.DataFrame(data)
    df.to_csv("kelulusan_mahasiswa.csv", index=False)
    print("File kelulusan_mahasiswa.csv dibuat!")

    # Langkah 2: Baca CSV
    df = pd.read_csv("kelulusan_mahasiswa.csv")
    print("\nInfo dataset:")
    print(df.info())
    print("\n5 baris pertama:")
    print(df.head())

    # Langkah 3: Cleaning
    print("\nCek missing values:")
    print(df.isnull().sum())
    df = df.drop_duplicates()
    print("\nSetelah hapus duplikat, jumlah data:", len(df))

    # Boxplot IPK untuk outlier
    sns.boxplot(x=df['IPK'])
    plt.title("Boxplot IPK")
    plt.show()

    # Langkah 4: EDA
    print("\nStatistik deskriptif:")
    print(df.describe())

    sns.histplot(df['IPK'], bins=10, kde=True)
    plt.title("Histogram Distribusi IPK")
    plt.show()

    sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
    plt.title("Scatterplot IPK vs Waktu Belajar")
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Heatmap Korelasi Fitur")
    plt.show()

    # Langkah 5: Feature Engineering
    df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
    df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
    df.to_csv("processed_kelulusan.csv", index=False)
    print("\nFile processed_kelulusan.csv berhasil dibuat!")

    # Langkah 6: Splitting dataset
    from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)
