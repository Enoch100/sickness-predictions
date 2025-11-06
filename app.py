
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import requests
import gdown
import pickle

st.set_page_config(page_title="Health Symptom Checker", layout="centered")
path="."
# Load model and metadata
#try:
#    model = joblib.load("./symptom_checker.pkl")
#    print("Model loaded successfully")
#except Exception as e:
#    print("Load failed:", e)

URL = "https://drive.google.com/uc?id=1645wfzc0VQ_lXzcOrXKhgvWM8EsdZcYf"
#URL = "https://drive.google.com/uc?id=1zPNJbJhH5SeKFB-JaKoA3DIkFeDBHUbA"
r=requests.get(URL)
output = "AAsymptom_checker.pkl"
output = "ssymptom_checker.pkl"
st.write("Downloading the model from google drive...")
gdown.download(URL, output, quiet=False)
with open(output, "rb") as f:
     #f.write(r.content)
     model = joblib.load(output)


# 2️⃣ Load the model
#model = joblib.load("AAsymptom_checker.pkl")
#model=joblib.load("symptom_checker.pkl", "wb").write(r.content)
#model = joblib.load("symptom_checker.pkl")
#le = joblib.load("ttps://github.com/Enoch100/sickness-prediction/blob/main/label_encoder.pkl")
#le = joblib.load("https://github.com/Enoch100/sickness-predictions/blob/main/label_encoder.pkl")

#import joblib

from io import BytesIO

url = "https://github.com/Enoch100/sickness-predictions/raw/main/label_encoder.pkl"
response = requests.get(url)
response.raise_for_status()  # ensure the request succeeded

le = joblib.load(BytesIO(response.content))


with open("./metadata.json", "r") as f:
    meta = json.load(f)

symptoms = meta["symptom_columns"]

# Load dataset for descriptions and precautions
df = pd.read_csv("./symptom_disease_full_41_final.csv")
df_meta = df[["disease","description","precaution","severity_level"]].drop_duplicates(subset=["disease"]).set_index("disease")

st.markdown("<h1 style='text-align:center; color: #2E8B57;'>🌿 Health Symptom Checker</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color: #2E8B57;'>Excellency</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#2E8B57;'>Select symptoms below and get the top 3 possible conditions (for educational use only).</p>", unsafe_allow_html=True)
st.write("---")

with st.expander("🔎 How it works (quick)"):
    st.write("This app uses a RandomForest model trained on a balanced symptom–disease dataset. It predicts possible conditions based on symptoms you select. This is for educational purposes and not a substitute for professional medical advice.")

# Multi-column layout for symptoms
cols = st.columns(3)
selected = []
for i, s in enumerate(symptoms):
    with cols[i % 3]:
        if st.checkbox(s.replace('_',' ').title(), key=s):
            selected.append(s)

if st.button("Check Possible Conditions"):
    if len(selected) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Build input vector
        x = np.zeros((1, len(symptoms)), dtype=int)
        for i, s in enumerate(symptoms):
            if s in selected:
                x[0,i] = 1
        # Predict probabilities
        probs = model.predict_proba(x)[0]
        top_idx = np.argsort(probs)[-3:][::-1]
        st.success("Top 3 possible conditions:")
        for rank_idx, idx_val in enumerate(top_idx, start=1):
            dis = le.inverse_transform([idx_val])[0]
            p = probs[idx_val]
            st.markdown("**{}{}. {}** — *{:.1f}% confidence*".format("", rank_idx, dis, p*100))
            # show metadata
            if dis in df_meta.index:
                info = df_meta.loc[dis]
                st.markdown("**Severity:** {}".format(info['severity_level'].title()))
                st.markdown("**Description:** {}".format(info['description']))
                st.markdown("**Precautions:** {}".format(info['precaution']))
            st.write("---")
        st.info("If symptoms are severe or worsening, seek emergency medical care immediately.")

st.markdown('<div style="position: fixed; bottom: 10px; right: 10px; color: #777;">Created with ❤️ — Educational tool</div>', unsafe_allow_html=True)


