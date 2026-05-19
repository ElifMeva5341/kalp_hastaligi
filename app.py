import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Sayfa Yapılandırması
st.set_page_config(page_title="Kalp Hastalığı Risk Tahmini", page_icon="❤️", layout="centered")

st.title("❤️ Kalp Hastalığı Risk Tahmin Sistemi (XAI Destekli)")
st.write("Lütfen hastanın klinik ve laboratuvar parametrelerini giriniz:")

# 1. Kaydedilen pkl dosyalarını yükleme
@st.cache_resource
def load_assets():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except FileNotFoundError:
    st.error("Lütfen 'rf_model.pkl', 'scaler.pkl' ve 'features.pkl' dosyalarını 'app.py' ile aynı klasöre koyun!")
    st.stop()

# 2. Dinamik Kullanıcı Girdileri (Seçilen özelliklere göre otomatik form oluşturur)
st.sidebar.header("📝 Hasta Parametreleri")
user_inputs = {}

# UCI Heart Disease veri setindeki olası özelliklerin Türkçe açıklamaları ve sınırları
feature_meta = {
    'age': ('Yaş', 29, 80, 50),
    'sex': ('Cinsiyet (0: Kadın, 1: Erkek)', [0, 1]),
    'cp': ('Göğüs Ağrısı Tipi (1: Tipik, 2: Atipik, 3: Anginal Olmayan, 4: Semptomsuz)', [1, 2, 3, 4]),
    'trestbps': ('Dinlenme Kan Basıncı (mm Hg)', 90, 200, 120),
    'chol': ('Serum Kolesterol (mg/dl)', 120, 560, 240),
    'fbs': ('Açlık Kan Şekeri > 120 mg/dl (0: Hayır, 1: Evet)', [0, 1]),
    'restecg': ('Dinlenme EKG Sonuçları (0, 1, 2)', [0, 1, 2]),
    'thalach': ('Maksimum Kalp Atış Hızı', 70, 210, 150),
    'exang': ('Egzersize Bağlı Anjin (0: Hayır, 1: Evet)', [0, 1]),
    'oldpeak': ('ST Depresyonu', 0.0, 6.2, 1.0),
    'slope': ('ST Segmenti Eğimi (1: Yükselen, 2: Düz, 3: İnen)', [1, 2, 3]),
    'ca': ('Renklendirilmiş Ana Damar Sayısı (0-3)', [0, 1, 2, 3]),
    'thal': ('Talyum Sintigrafisi (3: Normal, 6: Sabit Kusur, 7: Geri Dönüşümlü Kusur)', [3, 6, 7])
}

# Sadece SelectKBest tarafından seçilen özellikler için input arayüzü basılır
for feat in features:
    if feat in feature_meta:
        meta = feature_meta[feat]
        if isinstance(meta[1], list): # Kategorik değişkenler için selectbox
            user_inputs[feat] = st.sidebar.selectbox(meta[0], meta[1])
        else: # Sayısal değişkenler için slider
            if isinstance(meta[3], float):
                user_inputs[feat] = st.sidebar.slider(meta[0], float(meta[1]), float(meta[2]), float(meta[3]))
            else:
                user_inputs[feat] = st.sidebar.slider(meta[0], int(meta[1]), int(meta[2]), int(meta[3]))
    else:
        # Eğer beklenmedik bir özellik seçildiyse standart input basar
        user_inputs[feat] = st.sidebar.number_input(f"{feat}", value=0.0)

# Ana ekranda girilen değerleri gösterme
df_input = pd.DataFrame([user_inputs])
st.subheader("📋 Girilen Hasta Bilgileri")
st.dataframe(df_input)

# 3. Tahmin ve Açıklama Aşaması
if st.button("❤️ Risk Durumunu Tahmin Et ve Açıkla"):
    # Tam veri setine uygun sahte bir satır oluşturup, sadece seçilen özellikleri scale ediyoruz
    # Çünkü scaler nesnesi 13 özelliği eğiterek kaydedildi
    full_row = {f: 0.0 for f in feature_meta.keys()}
    for k, v in user_inputs.items():
        full_row[k] = v
    
    df_full = pd.DataFrame([full_row])
    # Colab'de eğittiğimiz scaler ile tüm satırı ölçeklendiriyoruz
    scaled_full = scaler.transform(df_full)
    df_scaled_full = pd.DataFrame(scaled_full, columns=feature_meta.keys())
    
    # Sadece seçilen alt özellikleri model girdisi olarak ayıklıyoruz
    X_input_scaled = df_scaled_full[features]
    
    # Model Tahminleri
    prediction = model.predict(X_input_scaled)[0]
    probability = model.predict_proba(X_input_scaled)[0][1]
    
    st.write("---")
    st.subheader("🔮 Model Çıktısı ve Risk Analizi")
    
    if prediction == 1:
        st.error(f"🚨 **Yüksek Kalp Hastalığı Riski!** (Olasılık: %{probability*100:.2f})")
    else:
        st.success(f"✅ **Düşük Kalp Hastalığı Riski.** (Olasılık: %{probability*100:.2f})")
        
    # 4. Kara Kutu Modelin Canlı Güvenilirlik Analizi (SHAP Waterfall) [cite: 37]
    st.write("---")
    st.subheader("🧬 Kara Kutu Modelin Karar Gerekçesi (SHAP Analizi)")
    st.write("Aşağıdaki grafik, girilen parametrelerin modelin verdiği kararı (riski) pozitif veya negatif yönde ne kadar etkilediğini gösterir[cite: 37].")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # SHAP boyut kontrolü (Binary classification esnekliği)
    if len(shap_values.shape) == 3:
        val_to_plot = shap_values.values[0, :, 1]
        base_val = shap_values.base_values[0, 1]
    else:
        val_to_plot = shap_values.values[0]
        base_val = shap_values.base_values[0]
        
    shap.plots.waterfall(
        shap.Explanation(
            values=val_to_plot,
            base_values=base_val,
            data=X_input_scaled.iloc[0],
            feature_names=features
        ),
        show=False
    )
    st.pyplot(plt.gcf())