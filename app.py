import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(page_title="MediGuide AI - Model 2", layout="wide")

BASE = os.path.dirname(__file__)

# ================================
# SIDEBAR
# ================================
st.sidebar.title("MediGuide AI")
st.sidebar.success("Model: TF-IDF")
st.sidebar.info("Free-text symptom prediction.")

# ================================
# HEADER
# ================================
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>MediGuide AI</h1>
<p style='text-align:center; color:gray;'>TF-IDF Disease Prediction</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# CLEAN
# ================================
def clean(t):
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t)

# ================================
# LOAD
# ================================
model = joblib.load(os.path.join(BASE, "symptoms_to_disease_model.pkl"))

with open(os.path.join(BASE, "precautions_map.pkl"), "rb") as f:
    prec_map = pickle.load(f)

desc_df = pd.read_csv(os.path.join(BASE, "symptom_Description.csv"))
desc_map = dict(zip(desc_df["Disease"].str.lower().str.replace(" ",""), desc_df["Description"]))

df = pd.read_csv(os.path.join(BASE, "DiseaseAndSymptoms.csv"))
symptoms = set()

for col in df.columns:
    if "Symptom" in col:
        symptoms.update(df[col].dropna().tolist())

symptom_list = sorted(symptoms)

# ================================
# INPUT UI
# ================================
col1, col2 = st.columns(2)

with col1:
    selected = st.multiselect("🧠 Select Symptoms", symptom_list)

with col2:
    text = st.text_area("✍️ Type Symptoms")

center = st.columns([1,2,1])
with center[1]:
    diagnose = st.button("🔍 Diagnose")

# ================================
# PREDICTION
# ================================
if diagnose:
    combined = " ".join(selected) + " " + text

    if not combined.strip():
        st.warning("Enter symptoms")
    else:
        probs = model.predict_proba([combined])[0]
        idx = np.argmax(probs)

        disease = model.classes_[idx]
        conf = probs[idx]

        key = disease.lower().replace(" ", "")

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric("Disease", disease)
        c2.metric("Confidence", f"{conf*100:.2f}%")
        c3.metric("Model", "TF-IDF")

        st.markdown("---")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 📄 Description")
            st.info(desc_map.get(key, "No description"))

        with colB:
            st.markdown("### 🛡️ Precautions")
            for p in prec_map.get(key, []):
                st.success(p)