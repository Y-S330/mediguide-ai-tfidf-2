import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(page_title="MediGuide AI", layout="wide")

BASE = os.path.dirname(__file__)

# ---------- STYLE ----------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .result-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- CLEAN ----------
def _clean(t):
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------- LOADERS ----------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "symptoms_to_disease_model.pkl"))

@st.cache_data
def load_precautions():
    with open(os.path.join(BASE, "precautions_map.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_descriptions():
    df = pd.read_csv(os.path.join(BASE, "symptom_Description.csv"))
    return dict(zip(df["Disease"].apply(_clean), df["Description"]))

@st.cache_data
def load_symptoms():
    df = pd.read_csv(os.path.join(BASE, "DiseaseAndSymptoms.csv"))
    syms = set()
    for col in df.columns:
        if "Symptom" in col:
            syms.update(df[col].dropna().str.strip().tolist())
    return sorted(syms)

# ---------- LOAD ----------
model = load_model()
prec_map = load_precautions()
desc_map = load_descriptions()
symptom_list = load_symptoms()

# ---------- PREDICT ----------
def predict_topk(inp, k=5):
    inp = _clean(inp)
    if not inp:
        return []

    proba = model.predict_proba([inp])[0]
    top_idx = np.argsort(proba)[::-1][:k]

    return [(model.classes_[i], float(proba[i])) for i in top_idx]

def get_precautions(name):
    return prec_map.get(_clean(name), [])

def get_description(name):
    return desc_map.get(_clean(name), "No description available")

# ---------- UI ----------
st.markdown('<p class="title">MediGuide AI - Disease Prediction System</p>', unsafe_allow_html=True)

st.markdown("### 🧠 Select or Enter Symptoms")

selected = st.multiselect("Select Symptoms", symptom_list)
text = st.text_area("Or type symptoms manually")

st.divider()

if st.button("Diagnose"):
    combined = " ".join(selected) + " " + text

    with st.spinner("Analyzing symptoms..."):
        results = predict_topk(combined)

    if not results:
        st.warning("Please enter at least one symptom.")
    else:
        disease, conf = results[0]

        st.markdown(f"""
        <div class="result-box">
        <h2 style='color:#4CAF50;'>Predicted Disease: {disease}</h2>
        <p><b>Confidence:</b> {conf*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📄 Description")
        st.info(get_description(disease))

        st.markdown("### 🛡️ Precautions")
        precautions = get_precautions(disease)

        if precautions:
            for p in precautions:
                st.success(p)
        else:
            st.warning("No precautions available.")