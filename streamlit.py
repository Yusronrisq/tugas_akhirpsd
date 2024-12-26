import streamlit as st
import numpy as np
from joblib import load

# Load model yang telah disimpan
model = load('rata-rata_knn.joblib')

# Judul Aplikasi
st.title("Aplikasi Klasifikasi Kelayakan Air Minum")

# Penjelasan Fitur
st.header("Penjelasan Fitur")
st.markdown("""
1. **pH**: Menunjukkan keseimbangan asam-basa air, dengan batas aman 6.5 hingga 8.5 (WHO).  
2. **Hardness**: Dipengaruhi oleh kalsium dan magnesium, menentukan kemampuan air membentuk busa sabun.  
3. **Solids (TDS)**: Total zat padat terlarut yang menunjukkan tingkat mineralisasi air. Batas maksimum 1000 mg/l.  
4. **Chloramines**: Disinfektan hasil penambahan amonia pada klorin, aman hingga 4 mg/L.  
5. **Sulfate**: Senyawa alami yang ada di tanah dan air, aman hingga 250 mg/L.  
6. **Conductivity**: Mengukur kemampuan air menghantarkan listrik, berkaitan dengan konsentrasi ion. Batas aman 400 µS/cm.  
7. **Organic Carbon**: Kandungan karbon organik total, batas aman < 2 mg/L di air minum.  
8. **Trihalomethanes (THMs)**: Zat kimia dari pengolahan klorin, aman hingga 80 ppm.  
9. **Turbidity**: Mengukur kekeruhan air akibat partikel tersuspensi, aman jika < 5 NTU.  
""")

# Input dari pengguna
st.header("Masukkan Nilai Fitur")
ph = st.number_input("pH", min_value=0.0, step=0.1, format="%.2f")
hardness = st.number_input("Hardness", min_value=0.0, step=0.1, format="%.2f")
solids = st.number_input("Solids (TDS)", min_value=0.0, step=1.0, format="%.2f")
chloramines = st.number_input("Chloramines", min_value=0.0, step=0.1, format="%.2f")
sulfate = st.number_input("Sulfate", min_value=0.0, step=0.1, format="%.2f")
conductivity = st.number_input("Conductivity", min_value=0.0, step=0.1, format="%.2f")
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, step=0.1, format="%.2f")
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, step=0.1, format="%.2f")
turbidity = st.number_input("Turbidity", min_value=0.0, step=0.1, format="%.2f")

# Tombol Prediksi
if st.button("Prediksi Kelayakan Air"):
    # Menggabungkan input fitur ke dalam array numpy
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, 
                            organic_carbon, trihalomethanes, turbidity]])
    
    # Prediksi menggunakan model
    prediction = model.predict(input_data)
    
    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        st.success("Hasil Prediksi: Air Layak Dikonsumsi (Potable)")
    else:
        st.error("Hasil Prediksi: Air Tidak Layak Dikonsumsi (Not Potable)")