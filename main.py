import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def mean_vectors(X_train, y_train):
    mean_values = {}
    for label in np.unique(y_train):  
        mean_values[label] = np.mean(X_train[y_train == label], axis=0)
    return mean_values

def classify(X_test, mean_values): 
    y_prediction = []
    for sample in X_test:
        min_distance = float('inf')
        predicted_label = None
        for label, mean_vector in mean_values.items():
            distance = np.linalg.norm(sample - mean_vector)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        y_prediction.append(predicted_label)
    return y_prediction


read_file = pd.read_csv('iris.csv')

loop = True
while(loop):
    print(" ")
    print("============ TUGAS 2 PKB ============")
    print("1. Tampilkan Data")
    print("2. Tambahkan Data Baru Kedalam Kolom")
    print("3. Klasifikasikan")
    print("4. Exit Program")
    print(" ")

    input_program = int(input("Silahkan Pilih: "))

    if(input_program == 1):
        tampilkan_data = read_file.head(len(read_file))
        print(" ")
        print("Berikut adalah datanya: ")
        print(" ")
        print(tampilkan_data)
        print(" ")
    elif(input_program == 2):
        new_sepallength = float(input("Masukkan nilai sepallength baru: "))
        new_sepalwidth = float(input("Masukkan nilai sepalwidth baru: "))
        new_petallength = float(input("Masukkan nilai petallength baru: "))
        new_petalwidth = float(input("Masukkan nilai petalwidth baru: "))

        new_data = {
            'sepallength': [new_sepallength],
            'sepalwidth': [new_sepalwidth],
            'petallength': [new_petallength],
            'petalwidth': [new_petalwidth],
            'class': ['']
        }

        new_data_frame = pd.DataFrame(new_data)
        read_file = pd.concat([read_file, new_data_frame], ignore_index=True)

        print(" ")
        print("Data telah ditambahkan")
        print(" ")
    elif(input_program == 3):
        X = read_file.iloc[:, :-1].values
        y = read_file['class'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)

        mean_values = mean_vectors(X_train, y_train)

        y_prediction = classify(X_test, mean_values)

        accuracy = accuracy_score(y_test, y_prediction)
        print(" ")
        print(f'Akurasi: {accuracy * 100:.2f}%')
        print(" ")
        print("Berikut adalah detail laporannya:")
        print(" ")
        print(classification_report(y_test, y_prediction))

        read_file.loc[len(read_file) - len(y_prediction):, 'class'] = y_prediction

        read_file.to_csv('iris.csv', index=False)
    elif(input_program == 4):
        loop = False
print(" ")
print("Terima Kasih!")