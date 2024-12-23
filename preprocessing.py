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
import numpy as np

# Streamlit configuration
st.title("Klasifikasi Kualitas Air")

# Sidebar configuration
menu = st.sidebar.selectbox(
    "Pilih Bagian",
    ("Deskripsi Dataset", "Imputasi Data", "Normalisasi & Penyeimbangan","Proses manual", "Evaluasi Model", "Perbandingan Model")
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

elif menu == "Proses manual":
    imputation_options = ["KNN", "Rata-rata"]
    selected_imputation = st.selectbox("Pilih Metode Imputasi", imputation_options)
    if selected_imputation == "KNN":
        X_resampled, y_resampled = X_knn_resampled, y_knn_resampled
    else:
        X_resampled, y_resampled = X_rata_resampled, y_rata_resampled
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    # Fungsi untuk menghitung Gini Index
    def gini_index(groups, classes):
        total_instances = sum([len(group) for group in groups])
        gini = 0.0

        for group in groups:
            size = len(group)
            if size == 0:
                continue

            score = 0.0
            for class_val in classes:
                proportion = [row[-1] for row in group].count(class_val) / size
                score += proportion ** 2

            gini += (1.0 - score) * (size / total_instances)

        return gini

    # Membagi data berdasarkan nilai split
    def split_data(index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Memilih split terbaik
    def get_best_split(dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None

        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = split_data(index, row[index], dataset)
                gini = gini_index(groups, class_values)
                st.write(f"Evaluating split: Feature[{index}], Value[{row[index]}], Gini[{gini}]")
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
                    st.write(f"Best split updated: Feature[{best_index}], Value[{best_value}], Gini[{best_score}]")

        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    # Membuat leaf node
    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Membagi node
    def split(node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])

        st.write(f"Splitting at depth {depth}: Left size[{len(left)}], Right size[{len(right)}]")

        if not left or not right:
            node['left'] = node['right'] = to_terminal(left + right)
            st.write(f"Leaf node created with value: {node['left']}")
            return

        if depth >= max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            st.write(f"Max depth reached. Creating leaf nodes: Left[{node['left']}], Right[{node['right']}]")
            return

        if len(left) <= min_size:
            node['left'] = to_terminal(left)
            st.write(f"Left group too small. Creating leaf node: {node['left']}")
        else:
            node['left'] = get_best_split(left)
            split(node['left'], max_depth, min_size, depth + 1)

        if len(right) <= min_size:
            node['right'] = to_terminal(right)
            st.write(f"Right group too small. Creating leaf node: {node['right']}")
        else:
            node['right'] = get_best_split(right)
            split(node['right'], max_depth, min_size, depth + 1)

    # Membangun decision tree
    def build_tree(train, max_depth, min_size):
        root = get_best_split(train)
        split(root, max_depth, min_size, 1)
        return root

    # Prediksi menggunakan decision tree
    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    data = data[:3]
    sample = [X_test[i] + [y_test[i]] for i in range(len(X_test))]

    # Membuat decision tree
    max_depth = 3
    min_size = 1
    tree = build_tree(data, max_depth, min_size)
    st.write("\nFinal Decision Tree:", tree)

    # Prediksi contoh
    result = predict(tree, sample)
    st.write("\nPrediction for sample:", result)
    y_pred = model.predict(X_test)
    # Evaluasi model
    st.markdown("### Evaluasi Model")
    st.write(f"Model: decesion tree dengan Imputasi: {selected_imputation}")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
elif menu == "Evaluasi Model":
    X_resampled, y_resampled = X_rata_resampled, y_rata_resampled
    st.write("## Splitting Data")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    st.write(f"Jumlah data training: {len(X_train)}")
    st.write(f"Jumlah data testing: {len(X_test)}")
    st.write("## Modeling data")
    model_options = ["Decision Tree", "Naive Bayes", "KNN"]
    imputation_options = ["KNN", "Rata-rata"]
    selected_model = st.selectbox("Pilih Metode Model", model_options)
    selected_imputation = st.selectbox("Pilih Metode Imputasi", imputation_options)
    if selected_imputation == "KNN":
        X_resampled, y_resampled = X_knn_resampled, y_knn_resampled
    else:
        X_resampled, y_resampled = X_rata_resampled, y_rata_resampled
    # Pemilihan model
    if selected_model == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_model == "Naive Bayes":
        model = GaussianNB()
    elif selected_model == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)
    # Training model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluasi model
    st.markdown("### Evaluasi Model")
    st.write(f"Model: {selected_model} dengan Imputasi: {selected_imputation}")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
elif menu == "Perbandingan Model":
    st.write(perbandingan)