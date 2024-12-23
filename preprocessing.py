import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit configuration
st.title("Klasifikasi Kualitas Air")

# Sidebar configuration
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Bagian",
    ("Deskripsi Dataset", "Imputasi Data", "Normalisasi & Penyeimbangan", "Evaluasi Model", "Perbandingan Model")
)

# Upload Dataset
data = pd.read_csv('water_potabilitys.csv')
imputer = KNNImputer(n_neighbors=10)
data_rata = data.fillna(data.mean())
data_KNN = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
scaler = MinMaxScaler()
X_knn = scaler.fit_transform(data_KNN.drop("Potability", axis=1))
X_rata = scaler.fit_transform(data_KNN.drop("Potability", axis=1))
y_knn = data_KNN["Potability"]
y_rata = data_KNN["Potability"]
smote = SMOTE()
X_knn_resampled, y_knn_resampled = smote.fit_resample(X_knn, y_knn)
X_rata_resampled, y_rata_resampled = smote.fit_resample(X_rata, y_rata)
X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
# Decision Tree Model dengan KNN
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred)
precision_1 = precision_score(y_test, y_pred)
recall_1 = recall_score(y_test, y_pred)
f1_1 = f1_score(y_test, y_pred)
#menggunakan decision tree dengan imputan rata-rata
X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_2 = accuracy_score(y_test, y_pred)
precision_2 = precision_score(y_test, y_pred)
recall_2 = recall_score(y_test, y_pred)
f1_2 = f1_score(y_test, y_pred)
#menggunakan naive bayes dengan imputan KNN
X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred)
precision_3 = precision_score(y_test, y_pred)
recall_3 = recall_score(y_test, y_pred)
f1_3 = f1_score(y_test, y_pred)

#menggunakan naive bayes dengan imputan rata-rata
X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_4 = accuracy_score(y_test, y_pred)
precision_4 = precision_score(y_test, y_pred)
recall_4 = recall_score(y_test, y_pred)
f1_4 = f1_score(y_test, y_pred)

#menggunakan KNN dengan imputan KNN
X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_5 = accuracy_score(y_test, y_pred)
precision_5 = precision_score(y_test, y_pred)
recall_5 = recall_score(y_test, y_pred)
f1_5 = f1_score(y_test, y_pred)
#menggunakan KNN dengan imputan rata-rata
X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_6 = accuracy_score(y_test, y_pred)
precision_6 = precision_score(y_test, y_pred)
recall_6 = recall_score(y_test, y_pred)
f1_6 = f1_score(y_test, y_pred)

accuracy_knn_decision_tree = round(accuracy_1,2)
precision_knn_decision_tree = round(precision_1,2)
recall_knn_decision_tree = round(recall_1,2)
f1_knn_decision_tree = round(f1_1,2)

accuracy_rata_decision_tree = round(accuracy_2,2)
precision_rata_decision_tree = round(precision_2,2)
recall_rata_decision_tree = round(recall_2,2)
f1_rata_decision_tree = round(f1_2,2)

accuracy_knn_naive_bayes = round(accuracy_3,2)
precision_knn_naive_bayes = round(precision_3,2)
recall_knn_naive_bayes = round(recall_3,2)
f1_knn_naive_bayes = round(f1_3,2)

accuracy_rata_naive_bayes = round(accuracy_4,2)
precision_rata_naive_bayes = round(precision_4,2)
recall_rata_naive_bayes = round(recall_4,2)
f1_rata_naive_bayes = round(f1_4,2)

accuracy_knn_KNN = round(accuracy_5,2)
precision_knn_KNN = round(precision_5,2)
recall_knn_KNN = round(recall_5,2)
f1_knn_KNN = round(f1_5,2)

accuracy_rata_KNN = round(accuracy_6,2)
precision_rata_KNN = round(precision_6,2)
recall_rata_KNN = round(recall_6,2)
f1_rata_KNN = round(f1_6,2)

hehe = {
    'Model': [
        'Decision Tree dengan Imputasi KNN',
        'Decision Tree dengan Imputasi Nilai Rata-rata',
        'Naive Bayes dengan Imputasi KNN',
        'Naive Bayes dengan Imputasi Nilai Rata-rata',
        'KNN dengan Imputasi KNN',
        'KNN dengan Imputasi Nilai Rata-rata'
    ],
    'Akurasi': [
        accuracy_knn_decision_tree,
        accuracy_rata_decision_tree,
        accuracy_knn_naive_bayes,
        accuracy_rata_naive_bayes,
        accuracy_knn_KNN,
        accuracy_rata_KNN
    ],
    'Precision': [
        precision_knn_decision_tree,
        precision_rata_decision_tree,
        precision_knn_naive_bayes,
        precision_rata_naive_bayes,
        precision_knn_KNN,
        precision_rata_KNN
    ],
    'Recall': [
        recall_knn_decision_tree,
        recall_rata_decision_tree,
        recall_knn_naive_bayes,
        recall_rata_naive_bayes,
        recall_knn_KNN,
        recall_rata_KNN
    ],
    'F1 Score': [
        f1_knn_decision_tree,
        f1_rata_decision_tree,
        f1_knn_naive_bayes,
        f1_rata_naive_bayes,
        f1_knn_KNN,
        f1_rata_KNN
    ]
}

perbandingan = pd.DataFrame(hehe)

if menu == "Deskripsi Dataset":
    st.write("### Dataset")
    st.write('''Dataset diperoleh dari Kaggle dengan nama ["water quality"](https://www.kaggle.com/datasets/adityakadiwal/water-potability). 
    Dataset ini berjumlah 3276 data dan memiliki 9 fitur dengan 1 target yaitu potability, berikut adalah fitur-fitur yang terdapat di dalamnya:
             
    - ph: Kandungan pH air (0 hingga 14).
    - Hardness: Kapasitas air untuk mengendapkan sabun dalam mg/L.
    - Solids: Total padatan terlarut dalam ppm.
    - Chloramines: Jumlah Kloramin dalam ppm.
    - Sulfate: Jumlah Sulfat yang terlarut dalam mg/L.
    - Conductivity: Konduktivitas listrik air dalam μS/cm.
    - Organic_carbon: Jumlah karbon organik dalam ppm.
    - Trihalomethanes: Jumlah Trihalomethanes dalam μg/L.
    - Turbidity: Ukuran sifat air yang memancarkan cahaya di NTU.
    - Potability: Menunjukkan apakah air aman untuk dikonsumsi manusia. Dapat diminum -1 dan Tidak dapat diminum -0
             ''')
    st.write(data.head())

elif menu == "Imputasi Data":
    st.markdown("## Handling Missing Values")
    st.write("Sebelum Imputasi:")
    st.write(data.isnull().sum())
    st.write("Karena data terdapat missing value, akan dilakukan pengisian menggunakan metode berikut:")
    imputer = KNNImputer(n_neighbors=5)
    data_rata = data.fillna(data.mean())
    data_KNN = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    st.write('''### Setelah Imputasi:
    1. Data setelah diisi menggunakan rata-rata:''')
    st.write(data_rata.head())
    st.write("2. Data setelah diisi menggunakan KNN:")
    st.write(data_KNN.head())
    st.write("dapat dilihat juga:")    
    st.write(data_KNN.isnull().sum())

elif menu == "Normalisasi & Penyeimbangan":
    st.markdown("## Normalisasi Data")
    scaler = MinMaxScaler()
    st.write("Data setelah normalisasi:")
    X_knn_normalized = pd.DataFrame(X_knn, columns=data.drop("Potability", axis=1).columns)
    X_rata_normalized = pd.DataFrame(X_knn, columns=data.drop("Potability", axis=1).columns)
    st.write(X_knn_normalized.head())
    st.write(X_rata_normalized.head())

    st.markdown("## Penyeimbangan Data")
    smote = SMOTE()
    st.write("kemudian data di seimbangkan menggunakan SMOTE, Distribusi setelah penyeimbangan:")
    st.write(pd.Series(y_knn_resampled).value_counts())

elif menu == "Evaluasi Model":
    st.write("## splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
    st.write(f"Jumlah data training: {len(X_train)}")
    st.write(f"Jumlah data testing: {len(X_test)}")
    #menggunakan decisiontree imputan KNN
    X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
    # Decision Tree Model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.markdown("### Evaluasi Model")
    st.markdown("### Model menggunakan decision tree dengan imputan KNN")
    accuracy_1 = accuracy_score(y_test, y_pred)
    precision_1 = precision_score(y_test, y_pred)
    recall_1 = recall_score(y_test, y_pred)
    f1_1 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_1:.2f}")
    st.write(f"Precision: {precision_1:.2f}")
    st.write(f"Recall: {recall_1:.2f}")
    st.write(f"F1 Score: {f1_1:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    #menggunakan decision tree dengan imputan rata
    X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
    st.markdown("### Model menggunakan decision tree dengan imputan rata-rata")
    # Decision Tree Model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy_2 = accuracy_score(y_test, y_pred)
    precision_2 = precision_score(y_test, y_pred)
    recall_2 = recall_score(y_test, y_pred)
    f1_2 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_2:.2f}")
    st.write(f"Precision: {precision_2:.2f}")
    st.write(f"Recall: {recall_2:.2f}")
    st.write(f"F1 Score: {f1_2:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    #menggunakan naive bayes dengan imputan KNN
    st.markdown("### Model menggunakan Naive bayes dengan imputan KNN")
    X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
    # Decision Tree Model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy_3 = accuracy_score(y_test, y_pred)
    precision_3 = precision_score(y_test, y_pred)
    recall_3 = recall_score(y_test, y_pred)
    f1_3 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_3:.2f}")
    st.write(f"Precision: {precision_3:.2f}")
    st.write(f"Recall: {recall_3:.2f}")
    st.write(f"F1 Score: {f1_3:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    #menggunakan naive bayes dengan imputan rata-rata
    X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
    # Decision Tree Model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.markdown("### Model menggunakan Naive bayes dengan imputan rata-rata")

    # Metrics
    accuracy_4 = accuracy_score(y_test, y_pred)
    precision_4 = precision_score(y_test, y_pred)
    recall_4 = recall_score(y_test, y_pred)
    f1_4 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_4:.2f}")
    st.write(f"Precision: {precision_4:.2f}")
    st.write(f"Recall: {recall_4:.2f}")
    st.write(f"F1 Score: {f1_4:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    #menggunakan KNN dengan imputan KNN
    X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
    st.markdown("### Model KNN bayes dengan imputan KNN")
    # Decision Tree Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_5 = accuracy_score(y_test, y_pred)
    precision_5 = precision_score(y_test, y_pred)
    recall_5 = recall_score(y_test, y_pred)
    f1_5 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_5:.2f}")
    st.write(f"Precision: {precision_5:.2f}")
    st.write(f"Recall: {recall_5:.2f}")
    st.write(f"F1 Score: {f1_5:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    #menggunakan KNN dengan imputan rata-rata
    X_train, X_test, y_train, y_test = train_test_split(X_rata_resampled, y_rata_resampled, test_size=0.2, random_state=42)
    st.markdown("### Model menggunakan KNN dengan imputan rata-rata")
    # Decision Tree Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy_6 = accuracy_score(y_test, y_pred)
    precision_6 = precision_score(y_test, y_pred)
    recall_6 = recall_score(y_test, y_pred)
    f1_6 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy_6:.2f}")
    st.write(f"Precision: {precision_6:.2f}")
    st.write(f"Recall: {recall_6:.2f}")
    st.write(f"F1 Score: {f1_6:.2f}")

elif menu == "Perbandingan Model":
    st.write(perbandingan)