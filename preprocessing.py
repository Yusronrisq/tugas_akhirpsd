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
X = data.drop(columns=['Potability'])
y = data['Potability']
imputer = KNNImputer(n_neighbors=10)
X_rata = X.fillna(X.mean())
X_KNN = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = MinMaxScaler(feature_range=(0, 1))
X_knn = pd.DataFrame(scaler.fit_transform(X_KNN), columns=X.columns)
X_rata = pd.DataFrame(scaler.fit_transform(X_rata), columns=X.columns)
smote = SMOTE(random_state=42)
X_knn_resampled, y_knn_resampled = smote.fit_resample(X_knn, y)
X_rata_resampled, y_rata_resampled = smote.fit_resample(X_rata, y)
X_train, X_test, y_train, y_test = train_test_split(X_knn_resampled, y_knn_resampled, test_size=0.2, random_state=42)
def evaluate_model(X, y, model, title, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Menampilkan confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title(title)
    plt.show()

    # Menampilkan hasil metrik evaluasi
    print(title)
    print(f'Akurasi: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    return accuracy, precision, recall, f1

# Data dan model untuk evaluasi
results = []
models = [
    # menggunakan train 90 dan test 10
    ("Decision Tree dengan KNN train 90", DecisionTreeClassifier(random_state=42), X_knn_resampled, y_knn_resampled, 0.1),
    ("Decision Tree dengan Rata-rata train 90", DecisionTreeClassifier(random_state=42), X_rata_resampled, y_rata_resampled, 0.1),
    ("Naive Bayes dengan KNN train 90", GaussianNB(), X_knn_resampled, y_knn_resampled, 0.1),
    ("Naive Bayes dengan Rata-rata train 90", GaussianNB(), X_rata_resampled, y_rata_resampled, 0.1),
    ("KNN dengan KNN train 90", KNeighborsClassifier(n_neighbors=3), X_knn_resampled, y_knn_resampled, 0.1),
    ("KNN dengan Rata-rata train 90", KNeighborsClassifier(n_neighbors=3), X_rata_resampled, y_rata_resampled, 0.1),
    # menggunakan train 80 dan test 20
    ("Decision Tree dengan KNN train 80", DecisionTreeClassifier(random_state=42), X_knn_resampled, y_knn_resampled, 0.2),
    ("Decision Tree dengan Rata-rata train 80", DecisionTreeClassifier(random_state=42), X_rata_resampled, y_rata_resampled, 0.2),
    ("Naive Bayes dengan KNN train 80", GaussianNB(), X_knn_resampled, y_knn_resampled, 0.2),
    ("Naive Bayes dengan Rata-rata train 80", GaussianNB(), X_rata_resampled, y_rata_resampled, 0.2),
    ("KNN dengan KNN train 80", KNeighborsClassifier(n_neighbors=3), X_knn_resampled, y_knn_resampled, 0.2),
    ("KNN dengan Rata-rata train 80", KNeighborsClassifier(n_neighbors=3), X_rata_resampled, y_rata_resampled, 0.2),
    # menggunakan train 70 dan test 30
    ("Decision Tree dengan KNN train 70", DecisionTreeClassifier(random_state=42), X_knn_resampled, y_knn_resampled, 0.3),
    ("Decision Tree dengan Rata-rata train 70", DecisionTreeClassifier(random_state=42), X_rata_resampled, y_rata_resampled, 0.3),
    ("Naive Bayes dengan KNN train 70", GaussianNB(), X_knn_resampled, y_knn_resampled, 0.3),
    ("Naive Bayes dengan Rata-rata train 70", GaussianNB(), X_rata_resampled, y_rata_resampled, 0.3),
    ("KNN dengan KNN train 70", KNeighborsClassifier(n_neighbors=3), X_knn_resampled, y_knn_resampled, 0.3),
    ("KNN dengan Rata-rata train 70", KNeighborsClassifier(n_neighbors=3), X_rata_resampled, y_rata_resampled, 0.3)
]
# Loop untuk evaluasi setiap model
for title, model, X, y, test_size in models:
    accuracy, precision, recall, f1 = evaluate_model(X, y, model, title, test_size)
    results.append((title, accuracy, precision, recall, f1))

perbandingan = pd.DataFrame(results, columns=["Model", "Akurasi", "Precision", "Recall", "F1 Score"])

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
    test_options = ['10','20','30']
    selected_test = st.selectbox("Pilih Data Test", test_options)
    if selected_imputation == "KNN":
        X_resampled, y_resampled = X_knn_resampled, y_knn_resampled
    else:
        X_resampled, y_resampled = X_rata_resampled, y_rata_resampled
    
    if selected_test == "10":
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    elif selected_test == "20":
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
        
    def gini_impurity(y):
        classes, counts = np.unique(y, return_counts=True)
        prob_squared = np.sum((counts / len(y)) ** 2)
        return 1 - prob_squared

    # Fungsi untuk membagi dataset berdasarkan threshold

    def split_dataset(X, y, feature, threshold):
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    # Fungsi untuk mencari split terbaik
    def best_split(X, y):
        n_features = X.shape[1]
        best_feature, best_threshold = None, None
        best_gini = float('inf')
        best_splits = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Hitung Gini Impurity
                gini_left = gini_impurity(y_left)
                gini_right = gini_impurity(y_right)

                gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (X_left, y_left, X_right, y_right)

        return best_feature, best_threshold, best_gini, best_splits

    # Definisi Node
    class DecisionNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    # Fungsi untuk membangun pohon keputusan

    def build_tree(X, y, depth=0, max_depth=None):
        # Kondisi berhenti: Semua label sama atau kedalaman maksimum tercapai
        if len(np.unique(y)) == 1 or (max_depth is not None and depth >= max_depth):
            return DecisionNode(value=np.bincount(y).argmax())

        feature, threshold, gini, splits = best_split(X, y)

        if feature is None or splits is None:
            return DecisionNode(value=np.bincount(y).argmax())

        X_left, y_left, X_right, y_right = splits
        st.write(f"kedalaman {depth}: membagi fitur {feature} dengan batas {threshold:.4f} dan Gini {gini:.4f}")

        # Rekursi ke subtree kiri dan kanan
        left_node = build_tree(X_left, y_left, depth + 1, max_depth)
        right_node = build_tree(X_right, y_right, depth + 1, max_depth)

        return DecisionNode(feature=feature, threshold=threshold, left=left_node, right=right_node)

    # Fungsi prediksi dari pohon keputusan
    def predict_tree(tree, X):
        if tree.value is not None:
            return tree.value
        
        if X[tree.feature] <= tree.threshold:
            return predict_tree(tree.left, X)
        else:
            return predict_tree(tree.right, X)

    # Fungsi prediksi untuk dataset
    def predict(dataset, tree):
        return np.array([predict_tree(tree, row) for row in dataset])
    
    def plot_tree(node, x=0.5, y=1, dx=0.25, dy=0.1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        if node.value is not None:
            ax.text(x, y, f"Leaf\nValue: {node.value}", ha='center', va='center',
                    bbox=dict(facecolor='lightblue', edgecolor='black'))
            return

        ax.text(x, y, f"Feature {node.feature}\n<= {node.threshold}", ha='center', va='center',
                bbox=dict(facecolor='lightgreen', edgecolor='black'))

        left_x, right_x = x - dx / 2, x + dx / 2
        ax.plot([x, left_x], [y, y - dy], color='black')
        ax.plot([x, right_x], [y, y - dy], color='black')

        plot_tree(node.left, x=left_x, y=y - dy, dx=dx / 2, dy=dy, ax=ax)
        plot_tree(node.right, x=right_x, y=y - dy, dx=dx / 2, dy=dy, ax=ax)

    # Pemanggilan untuk melatih dan memprediksi
    # Pastikan dataset terpisah dalam X_train, X_test, y_train, y_test seperti kode sebelumnya

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_test_np = X_test.to_numpy()

    decision_tree = build_tree(X_train_np, y_train_np, max_depth=3)
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(decision_tree,x=0.5, y=1, dx=0.1, dy=0.2, ax=ax)
    st.pyplot(fig)
    y_pred = predict(X_test_np, decision_tree)

    # Hitung akurasi
    accuracy = np.mean(y_pred == y_test.to_numpy())
    st.write(f"Accuracy: {accuracy:.2f}")

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
    test_options = ['10','20','30']
    selected_test = st.selectbox("Pilih data test", test_options)
    if selected_imputation == "KNN":
        X_resampled, y_resampled = X_knn_resampled, y_knn_resampled
    else:
        X_resampled, y_resampled = X_rata_resampled, y_rata_resampled
    
    if selected_test == "10":
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    elif selected_test == "20":
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
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
    metric = st.selectbox("Pilih metrik untuk ditampilkan:", ['Akurasi', 'Precision', 'Recall', 'F1 Score'])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(perbandingan['Model'], perbandingan[metric], alpha=0.7, color='skyblue')
    ax.set_title(f'Perbandingan {metric}', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticklabels(perbandingan['Model'], rotation=270, ha='center', fontsize=10)
    st.pyplot(fig)