import os
import re
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from html import escape

st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

# ==============================
# STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="block-container"] {
    padding: 2rem 3rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg,#38bdf8,#6366f1,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #94a3b8;
}
.stTextArea textarea {
    background: #020617 !important;
    border: 1.5px solid #1e293b !important;
    border-radius: 16px !important;
    transition: 0.2s;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.15) !important;
}
.stButton button {
    background: linear-gradient(135deg,#0ea5e9,#6366f1) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    letter-spacing: .5px;
    transition: .2s;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(14,165,233,.35);
}
.result-card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.result-card.top {
    border: 1.5px solid #38bdf8;
    background: linear-gradient(135deg, rgba(56,189,248,.18), rgba(99,102,241,.08));
    box-shadow: 0 20px 60px rgba(56,189,248,.22);
}
.disease-name {
    font-size: 1.45rem;
    font-weight: 900;
}
.bar-bg {
    background: #1e293b;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
}
.bar {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg,#38bdf8,#6366f1);
}
.section-label {
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #cbd5e1;
}
.about-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1rem;
    margin-top: 1rem;
}
.treatment-box {
    background: linear-gradient(135deg, rgba(14,165,233,.08), rgba(14,165,233,.02));
    border: 1px solid rgba(14,165,233,.2);
    border-radius: 14px;
    padding: 1rem;
    margin-top: 1rem;
}
.prec-card {
    background: linear-gradient(135deg, rgba(52,211,153,.08), rgba(52,211,153,.02));
    border: 1px solid rgba(52,211,153,.2);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
}
.warn-box {
    background: rgba(251,191,36,.08);
    border: 1px solid rgba(251,191,36,.18);
    color: #fcd34d;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-top: 1rem;
}
.low-conf {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.18);
    color: #fca5a5;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.footer {
    color: #334155;
    opacity: .75;
    margin-top: 2rem;
}
.small-muted {
    color: #64748b;
    font-size: .82rem;
}
.metric-pill {
    display: inline-block;
    background: rgba(56,189,248,.08);
    border: 1px solid rgba(56,189,248,.18);
    color: #7dd3fc;
    border-radius: 999px;
    padding: .28rem .7rem;
    font-size: .78rem;
    margin-right: .4rem;
    margin-bottom: .4rem;
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

# ==============================
# CURATED TREATMENT FALLBACK
# ==============================
TREATMENT_FALLBACK = {
    "malaria": "Treatment involves antimalarial medications such as artemisinin-based combination therapies (ACTs) or chloroquine depending on the Plasmodium species. Seek immediate medical care. Rest, hydration, and paracetamol for fever are supportive measures. Never self-medicate.",
    "dengue": "No specific antiviral exists. Treatment is supportive: rest, oral rehydration, and paracetamol for fever — avoid aspirin and ibuprofen as they increase bleeding risk. Monitor for warning signs like severe abdominal pain, bleeding, or rapid breathing and seek emergency care immediately.",
    "chicken pox": "Treatment is mainly supportive: calamine lotion for itching, paracetamol for fever, and keeping nails trimmed to prevent scratching. Antiviral acyclovir may be prescribed for high-risk patients. Keep the patient isolated to prevent spread.",
    "drug reaction": "Stop the suspected medication immediately and consult a doctor or go to the nearest hospital. For mild reactions, antihistamines or corticosteroids may be prescribed. Severe reactions such as anaphylaxis require emergency epinephrine.",
    "allergy": "Treatment includes antihistamines, decongestants, or corticosteroids depending on severity. Severe allergic reactions require emergency epinephrine. Long-term relief can come from allergen immunotherapy prescribed by an allergist. Avoid known triggers.",
    "acne": "Topical retinoids, benzoyl peroxide, or salicylic acid may help mild cases. Moderate-to-severe acne may require oral antibiotics, hormonal therapy, or isotretinoin prescribed by a dermatologist. Keep skin clean and avoid squeezing pimples.",
    "gerd": "Lifestyle changes help first: avoid trigger foods, eat smaller meals, and do not lie down after eating. Medications include antacids, H2 blockers, or proton pump inhibitors. Consult a doctor if symptoms persist.",
    "bronchial asthma": "Reliever inhalers such as salbutamol help immediate symptoms. Preventer inhalers such as corticosteroids help long-term control. Avoid known triggers. In severe attacks seek emergency care immediately.",
    "typhoid": "Typhoid is treated with antibiotics prescribed by a doctor. Rest and adequate fluid intake are essential. Complete the full antibiotic course even if you feel better.",
    "common cold": "There is no cure. Treatment is supportive: rest, hydration, saline rinses, and paracetamol or ibuprofen for fever and aches. Antibiotics are not effective for viral colds.",
    "gastroenteritis": "Oral rehydration is the most important step. Rest and bland foods can help. Seek care immediately if dehydration becomes severe.",
    "heart attack": "Emergency. Call an ambulance immediately. While waiting, chew aspirin if not allergic and if a clinician has not told you otherwise. Hospital treatment is urgent and time-sensitive.",
    "urinary tract infection": "Antibiotics are usually needed based on urine findings. Drink fluids and complete the full course exactly as prescribed.",
    "migraine": "Acute treatment may include triptans, NSAIDs, paracetamol, or antiemetics. Rest in a quiet, dark room and avoid known triggers.",
    "hypertension": "Lifestyle changes matter: low-sodium diet, regular exercise, weight control, and limiting alcohol. Medicines may include ACE inhibitors, ARBs, calcium channel blockers, or diuretics.",
    "diabetes": "Type 1 diabetes needs insulin. Type 2 diabetes usually starts with lifestyle changes and metformin, with other medicines added if needed. Regular blood glucose monitoring is important.",
    "tuberculosis": "Tuberculosis requires a full multi-drug antibiotic course. Never stop early because incomplete treatment can cause drug resistance.",
    "pneumonia": "Bacterial pneumonia is often treated with antibiotics. Rest, fluids, and monitoring are important. Severe cases may need oxygen or hospital care.",
    "hypothyroidism": "Levothyroxine is taken daily and the dose is adjusted using thyroid blood tests. Long-term follow-up is usually needed.",
    "hyperthyroidism": "Treatment may include antithyroid medicines, radioactive iodine, or surgery. Beta-blockers may help symptoms while waiting for definitive treatment.",
    "hypoglycemia": "Immediate treatment is 15–20 g of fast-acting carbohydrate such as glucose tablets or juice. Recheck after 15 minutes. Severe cases need urgent medical help.",
    "psoriasis": "Topical treatments help mild cases. Phototherapy or systemic medicines may be needed for more severe disease. Regular moisturising also helps.",
    "arthritis": "Treatment depends on type. It may include pain relief, physiotherapy, disease-modifying drugs, or other targeted treatment.",
    "aids": "Antiretroviral therapy is essential. Early treatment and regular monitoring greatly improve long-term outcomes.",
    "peptic ulcer diseae": "Treatment depends on cause. H. pylori ulcers are often treated with antibiotics plus a proton pump inhibitor. NSAID-related ulcers are treated by stopping the NSAID if possible and using acid suppression."
}

# ==============================
# HELPERS
# ==============================
def _clean(text):
    text = str(text).lower().strip()
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _safe_html_text(text):
    return escape(str(text)).replace("\n", "<br>")

def _shorten_text(text, max_chars=450):
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = cut.rfind(".")
    return cut[:last + 1] if last > 220 else cut.rstrip() + "..."

def _tokenize(text):
    return [tok for tok in _clean(text).split() if len(tok) >= 3]

def _match_lookup(key, mapping):
    key = _clean(key)
    if key in mapping:
        return mapping[key]
    for k in mapping:
        if k in key or key in k:
            return mapping[k]
    return None

# ==============================
# LOADERS
# ==============================
@st.cache_resource
def load_model():
    path = os.path.join(BASE, "symptoms_to_disease_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "symptoms_to_disease_model.pkl not found. Run the notebook first and place it beside app.py."
        )
    return joblib.load(path)

@st.cache_resource
def load_model_lr():
    path = os.path.join(BASE, "model_lr.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_model_cnb():
    path = os.path.join(BASE, "model_cnb.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_precautions():
    path = os.path.join(BASE, "precautions_map.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}

@st.cache_data
def load_descriptions():
    path = os.path.join(BASE, "symptom_Description.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    required = {"Disease", "Description"}
    if not required.issubset(df.columns):
        return {}

    df = df.copy()
    df["Disease"] = df["Disease"].astype(str).apply(_clean)
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df = df[(df["Disease"] != "") & (df["Description"] != "")]
    return dict(zip(df["Disease"], df["Description"]))

@st.cache_data
def load_medquad():
    pkl_path = os.path.join(BASE, "medquad_df.pkl")
    csv_path = os.path.join(BASE, "medquad.csv")

    if os.path.exists(pkl_path):
        data = joblib.load(pkl_path)
        if isinstance(data, pd.DataFrame):
            return data.copy()

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        if isinstance(data, pd.DataFrame):
            return data.copy()

    return None

@st.cache_data
def load_symptoms_list():
    path = os.path.join(BASE, "DiseaseAndSymptoms.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    symptom_cols = [c for c in df.columns if "symptom" in c.lower()]
    if not symptom_cols:
        return []

    vals = set()
    for col in symptom_cols:
        series = df[col].dropna().astype(str).str.strip()
        vals.update(v for v in series if v)

    return sorted(vals)

# ==============================
# RESOURCE LOAD
# ==============================
try:
    model = load_model()
    model_lr = load_model_lr()
    model_cnb = load_model_cnb()
    prec_map = load_precautions()
    desc_map = load_descriptions()
    medquad_df = load_medquad()
    symptom_list = load_symptoms_list()
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error(
        "❌ The loaded main model does not support probability estimates. "
        "Use your calibrated symptoms_to_disease_model.pkl."
    )
    st.stop()

_use_ensemble = (model_lr is not None) and (model_cnb is not None)

# ==============================
# SYMPTOM NORMALIZATION
# ==============================
SYMPTOM_MAP = {
    "high fever": "high_fever", "fever": "high_fever", "very high fever": "high_fever",
    "mild fever": "mild_fever", "slight fever": "mild_fever", "low grade fever": "mild_fever",
    "headache": "headache", "head ache": "headache", "migraine": "headache",
    "cough": "cough", "dry cough": "cough", "wet cough": "cough", "coughing": "cough",
    "shortness of breath": "breathlessness", "difficulty breathing": "breathlessness", "breathlessness": "breathlessness",
    "chest pain": "chest_pain", "chest discomfort": "chest_pain",
    "stomach pain": "stomach_pain", "stomach ache": "stomach_pain", "abdominal pain": "abdominal_pain",
    "body ache": "muscle_pain", "body pain": "muscle_pain", "muscle pain": "muscle_pain",
    "joint pain": "joint_pain", "joint ache": "joint_pain",
    "runny nose": "runny_nose", "blocked nose": "congestion", "nasal congestion": "congestion", "congestion": "congestion",
    "sore throat": "throat_irritation", "throat pain": "throat_irritation",
    "phlegm": "phlegm", "mucus": "phlegm", "sneezing": "continuous_sneezing",
    "fatigue": "fatigue", "tired": "fatigue", "tiredness": "fatigue", "weakness": "fatigue",
    "lethargy": "lethargy", "no energy": "lethargy",
    "nausea": "nausea", "vomiting": "vomiting", "vomit": "vomiting", "throwing up": "vomiting",
    "diarrhea": "diarrhoea", "diarrhoea": "diarrhoea", "loose stool": "diarrhoea",
    "constipation": "constipation", "heartburn": "acidity", "acidity": "acidity", "indigestion": "indigestion",
    "loss of appetite": "loss_of_appetite", "no appetite": "loss_of_appetite",
    "bloating": "distention_of_abdomen", "bloated": "distention_of_abdomen",
    "gas": "passage_of_gases", "flatulence": "passage_of_gases",
    "itching": "itching", "itch": "itching", "itchy": "itching",
    "rash": "skin_rash", "skin rash": "skin_rash", "rashes": "skin_rash",
    "blister": "blister", "blisters": "blister",
    "yellow eyes": "yellowing_of_eyes", "yellowing of eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin", "yellowish skin": "yellowish_skin",
    "red eyes": "redness_of_eyes", "watery eyes": "watering_from_eyes",
    "blurred vision": "blurred_and_distorted_vision", "blurry vision": "blurred_and_distorted_vision",
    "dizziness": "dizziness", "dizzy": "dizziness", "lightheaded": "dizziness",
    "chills": "chills", "shivering": "shivering", "sweating": "sweating", "night sweats": "sweating",
    "palpitations": "palpitations", "racing heart": "fast_heart_rate", "fast heart rate": "fast_heart_rate",
    "burning urination": "burning_micturition", "pain urinating": "burning_micturition",
    "frequent urination": "polyuria",
    "dark urine": "dark_urine", "yellow urine": "yellow_urine",
    "swollen legs": "swollen_legs", "swollen joints": "swelling_joints", "swelling": "swelling_joints",
    "stiff neck": "stiff_neck", "joint stiffness": "movement_stiffness",
    "anxiety": "anxiety", "anxious": "anxiety", "depression": "depression", "depressed": "depression",
    "mood swings": "mood_swings", "irritability": "irritability",
    "loss of balance": "loss_of_balance", "slurred speech": "slurred_speech",
    "dehydration": "dehydration", "thirsty": "dehydration",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "enlarged thyroid": "enlarged_thyroid",
}

_SORTED_PHRASES = sorted(SYMPTOM_MAP.keys(), key=len, reverse=True)

def normalize_free_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[,;]+", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    found = []
    remaining = text

    for phrase in _SORTED_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, remaining):
            mapped = SYMPTOM_MAP[phrase]
            if mapped not in found:
                found.append(mapped)
            remaining = re.sub(pattern, " ", remaining)

    leftover = re.sub(r"\s+", " ", remaining).strip()
    if leftover:
        found.append(leftover)

    return " ".join(found) if found else text

def count_recognized(text):
    text = str(text).lower()
    text = re.sub(r"[,;]+", " ", text)
    remaining = text
    count = 0

    for phrase in _SORTED_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, remaining):
            count += 1
            remaining = re.sub(pattern, " ", remaining)

    return count

# ==============================
# RETRIEVAL / LOOKUPS
# ==============================
def retrieve_treatment_from_medquad(predicted_disease, top_k=2, min_token_hits=2):
    if medquad_df is None or not isinstance(medquad_df, pd.DataFrame):
        return []

    disease_clean = _clean(predicted_disease)
    if not disease_clean:
        return []

    tokens = _tokenize(disease_clean)
    if not tokens:
        return []

    df = medquad_df.copy()

    if "question_clean" in df.columns:
        df["q_clean"] = df["question_clean"].fillna("").astype(str).apply(_clean)
    elif "question" in df.columns:
        df["q_clean"] = df["question"].fillna("").astype(str).apply(_clean)
    else:
        return []

    if "answer_clean" in df.columns:
        df["a_clean"] = df["answer_clean"].fillna("").astype(str)
        df["a_search"] = df["answer_clean"].fillna("").astype(str).apply(_clean)
    elif "answer" in df.columns:
        df["a_clean"] = df["answer"].fillna("").astype(str)
        df["a_search"] = df["answer"].fillna("").astype(str).apply(_clean)
    else:
        return []

    df = df[(df["q_clean"] != "") & (df["a_clean"].str.strip() != "")]
    if df.empty:
        return []

    phrase_pat = rf"\b{re.escape(disease_clean)}\b"
    phrase_in_q = df["q_clean"].str.contains(phrase_pat, regex=True, na=False)
    phrase_in_a = df["a_search"].str.contains(phrase_pat, regex=True, na=False)

    effective_min_hits = 1 if len(tokens) == 1 else int(min_token_hits)
    tok_pat = r"\b(" + "|".join(map(re.escape, tokens)) + r")\b"

    tok_in_q = df["q_clean"].str.findall(tok_pat).str.len()
    tok_in_a = df["a_search"].str.findall(tok_pat).str.len()

    cand_mask = (phrase_in_q | phrase_in_a) | (tok_in_q >= effective_min_hits) | (tok_in_a >= effective_min_hits)
    cands = df[cand_mask].copy()

    if cands.empty:
        return []

    cands["_phrase_q"] = phrase_in_q[cand_mask].astype(int).values
    cands["_phrase_a"] = phrase_in_a[cand_mask].astype(int).values
    cands["_tok_q"] = tok_in_q[cand_mask].values
    cands["_tok_a"] = tok_in_a[cand_mask].values

    cands["score"] = (
        3 * cands["_phrase_q"] +
        3 * cands["_phrase_a"] +
        1 * cands["_tok_q"] +
        1 * cands["_tok_a"]
    )
    cands["ans_len"] = cands["a_clean"].astype(str).str.len()
    cands = cands.sort_values(["score", "ans_len"], ascending=[False, True])

    answers = []
    seen = set()

    for ans in cands["a_clean"].tolist():
        cleaned_ans = str(ans).strip()
        if cleaned_ans and cleaned_ans not in seen:
            seen.add(cleaned_ans)
            answers.append(_shorten_text(cleaned_ans))
        if len(answers) >= int(top_k):
            break

    return answers

def get_precautions(disease_name):
    out = _match_lookup(disease_name, prec_map)
    return out if isinstance(out, list) else []

def get_description(disease_name):
    return _match_lookup(disease_name, desc_map)

def get_treatment(disease_name):
    fallback = _match_lookup(disease_name, TREATMENT_FALLBACK)
    if fallback:
        return [fallback]

    mq = retrieve_treatment_from_medquad(disease_name, top_k=2)
    if mq:
        return mq

    return []

# ==============================
# PREDICTION
# ==============================
def predict_topk(inp, k=5):
    inp = normalize_free_text(inp)
    inp = _clean(inp)

    if not inp:
        return []

    k = max(1, int(k))

    if _use_ensemble:
        weights = (0.5, 0.3, 0.2)
        all_classes = model.classes_
        n = len(all_classes)
        k = min(k, n)
        class_idx = {c: i for i, c in enumerate(all_classes)}

        def align_proba(m):
            p = m.predict_proba([inp])[0]
            aligned = np.zeros(n, dtype=float)
            for i, cls in enumerate(m.classes_):
                idx = class_idx.get(cls)
                if idx is not None:
                    aligned[idx] = float(p[i])
            return aligned

        blended = (
            weights[0] * align_proba(model) +
            weights[1] * align_proba(model_lr) +
            weights[2] * align_proba(model_cnb)
        )

        top_idx = np.argsort(blended)[::-1][:k]
        return [(all_classes[i], float(blended[i])) for i in top_idx]

    proba = model.predict_proba([inp])[0]
    k = min(k, len(model.classes_))
    top_idx = np.argsort(proba)[::-1][:k]
    return [(model.classes_[i], float(proba[i])) for i in top_idx]

def confidence_level(top5):
    if not top5:
        return "none", 0.0

    top1 = float(top5[0][1])
    top2 = float(top5[1][1]) if len(top5) > 1 else 0.0
    margin = top1 - top2

    if top1 >= 0.40:
        return "high", margin
    if top1 >= 0.33 and margin >= 0.12:
        return "medium", margin
    if top1 >= 0.05:
        return "low", margin
    return "none", margin

# ==============================
# STATE
# ==============================
if "selected_syms" not in st.session_state:
    st.session_state["selected_syms"] = []
if "free_text" not in st.session_state:
    st.session_state["free_text"] = ""

# ==============================
# UI
# ==============================
st.markdown(
    '<div class="hero"><h1>🩺 MediGuide AI</h1>'
    '<p>Describe your symptoms and get AI-powered disease insights, treatment information, and precautions.</p></div>',
    unsafe_allow_html=True
)

if _use_ensemble:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1rem">'
        '<span class="metric-pill">⚡ Ensemble Mode: LinearSVC + Logistic Regression + ComplementNB</span>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1rem">'
        '<span class="metric-pill">⚠️ Single Model Mode</span>'
        '</div>',
        unsafe_allow_html=True
    )

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Enter Symptoms</div>', unsafe_allow_html=True)

    selected_syms = st.multiselect(
        "Choose from known symptoms",
        options=symptom_list,
        format_func=lambda x: x.replace("_", " ").title(),
        placeholder="Search symptoms...",
        key="selected_syms"
    )

    free_text = st.text_area(
        "Or describe symptoms in free text (English)",
        placeholder="e.g. high fever headache chills sweating nausea",
        height=120,
        key="free_text"
    )

    combined_text = " ".join(selected_syms) + " " + free_text
    recognized_count = count_recognized(free_text)
    has_dropdown_symptoms = len(selected_syms) > 0

    c1, c2 = st.columns([2, 1])
    with c1:
        run = st.button("🔍 Diagnose", use_container_width=True)
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state["selected_syms"] = []
            st.session_state["free_text"] = ""
            st.rerun()

    if free_text.strip():
        remaining = free_text.lower()
        remaining = re.sub(r"[,;]+", " ", remaining)
        remaining = re.sub(r"\band\b", " ", remaining)
        recognized_pills = []

        for phrase in _SORTED_PHRASES:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, remaining):
                recognized_pills.append(phrase)
                remaining = re.sub(pattern, " ", remaining)

        if recognized_pills:
            pills_html = "".join(
                f'<span class="metric-pill">✓ {escape(p)}</span>'
                for p in recognized_pills
            )
            st.markdown(
                f'<div class="section-label">Recognized symptoms</div>{pills_html}',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="warn-box">⚠️ No known symptoms recognized yet. Try words like fever, headache, cough, chest pain, nausea, fatigue, vomiting, chills, sweating, or dizziness.</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-label">Tips</div>'
        '<div class="about-box">'
        'Use symptoms from the same illness only. Mixing unrelated symptoms can lower confidence.<br><br>'
        '<span class="small-muted">Commas are optional. Uppercase and lowercase both work.</span>'
        '</div>',
        unsafe_allow_html=True
    )

with right:
    st.markdown('<div class="section-label">Diagnosis Results</div>', unsafe_allow_html=True)

    if run:
        if not combined_text.strip():
            st.warning("⚠️ Please enter at least one symptom.")
        elif not has_dropdown_symptoms and free_text.strip() and recognized_count == 0:
            st.warning("⚠️ No known symptoms were recognized. Try clearer symptom words or use the dropdown list.")
        else:
            with st.spinner("Analyzing symptoms..."):
                results = predict_topk(combined_text, k=5)

            if not results:
                st.error("❌ Could not process input.")
            else:
                level, margin = confidence_level(results)
                top_disease, top_conf = results[0]

                if level == "none":
                    st.markdown(
                        '<div class="low-conf"><b>Could not identify a condition.</b><br>'
                        'Try adding more specific symptoms such as fever, cough, headache, nausea, or chest pain.</div>',
                        unsafe_allow_html=True
                    )

                elif level == "low":
                    st.markdown(
                        '<div class="low-conf"><b>Low confidence prediction.</b><br>'
                        'The model found some signal, but it is weak. Add more symptoms for a better result.</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="metric-pill">Top confidence: {top_conf * 100:.1f}%</div>'
                        f'<div class="metric-pill">Margin: {margin:.3f}</div>',
                        unsafe_allow_html=True
                    )

                    for disease, conf in results[:3]:
                        bar_w = min(conf * 100, 100)
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="disease-name" style="font-size:1.08rem">{escape(disease.title())}</div>
                            <div class="bar-bg"><div class="bar" style="width:{bar_w:.0f}%;opacity:.6"></div></div>
                            <div class="small-muted">{conf * 100:.1f}% confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    desc = get_description(top_disease)
                    treatments = get_treatment(top_disease)
                    precautions = get_precautions(top_disease)
                    bar_w = min(top_conf * 100, 100)

                    if level == "medium":
                        st.markdown(
                            '<div class="warn-box">⚠️ Moderate confidence. Your symptoms also overlap with other conditions. Adding more symptom details may improve the result.</div>',
                            unsafe_allow_html=True
                        )

                    about_html = ""
                    if desc:
                        about_html = (
                            '<div class="about-box">'
                            '<div class="section-label" style="margin-top:0">About this condition</div>'
                            f'{_safe_html_text(desc)}'
                            '</div>'
                        )

                    treatment_html = ""
                    if treatments:
                        items = "".join(
                            f'<div style="margin-bottom:.6rem">{_safe_html_text(t)}</div>'
                            for t in treatments
                        )
                        treatment_html = (
                            '<div class="treatment-box">'
                            '<div class="section-label" style="margin-top:0">💊 Treatment information</div>'
                            f'{items}'
                            '</div>'
                        )

                    st.markdown(f"""
                    <div class="result-card top">
                        <div class="small-muted">Top diagnosis</div>
                        <div class="disease-name">{escape(top_disease.title())}</div>
                        <div class="bar-bg"><div class="bar" style="width:{bar_w:.0f}%"></div></div>
                        <div style="display:flex;justify-content:space-between;margin-top:.45rem">
                            <span class="small-muted">Confidence</span>
                            <span>{top_conf * 100:.1f}%</span>
                        </div>
                        <div style="margin-top:.65rem">
                            <span class="metric-pill">Decision margin: {margin:.3f}</span>
                            <span class="metric-pill">Confidence level: {level.title()}</span>
                        </div>
                        {about_html}
                        {treatment_html}
                    </div>
                    """, unsafe_allow_html=True)

                    if precautions:
                        st.markdown('<div class="section-label">Recommended precautions</div>', unsafe_allow_html=True)
                        for i, p in enumerate(precautions, start=1):
                            st.markdown(
                                f'<div class="prec-card"><b>{i}.</b> {_safe_html_text(p)}</div>',
                                unsafe_allow_html=True
                            )

                    if len(results) > 1:
                        st.markdown('<div class="section-label">Other possibilities</div>', unsafe_allow_html=True)
                        for disease, conf in results[1:4]:
                            bar_w = min(conf * 100, 100)
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="disease-name" style="font-size:1.05rem">{escape(disease.title())}</div>
                                <div class="bar-bg"><div class="bar" style="width:{bar_w:.0f}%;opacity:.65"></div></div>
                                <div class="small-muted">{conf * 100:.1f}% confidence</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown(
                        '<div class="warn-box">⚕️ AI-generated educational result. It is not a medical diagnosis. Always consult a qualified doctor, especially if symptoms are severe, sudden, or worsening.</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.markdown(
            '<div style="text-align:center;padding:4rem 1rem;color:#334155">'
            '<div style="font-size:3rem">🩺</div>'
            '<div style="font-size:1.05rem;margin-top:1rem">Enter your symptoms and click Diagnose</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown('<div class="footer">MediGuide AI • For educational purposes only</div>', unsafe_allow_html=True)