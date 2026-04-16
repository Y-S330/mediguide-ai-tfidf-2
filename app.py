import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* 🔥 FIX SPACING (MAJOR IMPROVEMENT) */
[data-testid="block-container"] {
    padding: 2rem 3rem;
}

/* HERO */
.hero h1 {
    font-size:3.2rem;
    font-weight:900;
    background:linear-gradient(135deg,#38bdf8,#6366f1,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p {
    color:#94a3b8;
}

/* INPUT */
.stTextArea textarea {
    background:#020617!important;
    border:1.5px solid #1e293b!important;
    border-radius:16px!important;
    transition:0.2s;
}
.stTextArea textarea:focus {
    border-color:#38bdf8!important;
    box-shadow:0 0 0 3px rgba(56,189,248,.15)!important;
}

/* BUTTON */
.stButton button {
    background:linear-gradient(135deg,#0ea5e9,#6366f1)!important;
    border-radius:14px!important;
    font-weight:600!important;
    letter-spacing:.5px;
    transition:.2s;
}
.stButton button:hover {
    transform:translateY(-2px);
    box-shadow:0 12px 30px rgba(14,165,233,.35);
}

/* CARDS */
.result-card {
    background:#020617;
    border:1px solid #1e293b;
    border-radius:18px;
    padding:1.6rem;
    margin-bottom:1.4rem;
    transition:0.25s ease;
}
.result-card:hover {
    transform:translateY(-5px) scale(1.01);
    box-shadow:0 15px 40px rgba(0,0,0,.6);
}

/* 🔥 TOP RESULT (STRONGER) */
.result-card.top {
    border:1.5px solid #38bdf8;
    background:linear-gradient(135deg, rgba(56,189,248,.20), rgba(99,102,241,.08));
    box-shadow:0 25px 80px rgba(56,189,248,.30);
    transform:scale(1.03);
}

/* TEXT */
.disease-name {
    font-size:1.6rem;
    font-weight:900;
}

/* BAR */
.bar-bg {
    background:#1e293b;
    height:7px;
    border-radius:99px;
}
.bar {
    height:7px;
    border-radius:99px;
    background:linear-gradient(90deg,#38bdf8,#6366f1);
}

/* SECTION SPACING */
.section-label {
    margin-top:1.8rem;
    margin-bottom:0.6rem;
}

/* ABOUT */
.about-box {
    background:#020617;
    border:1px solid #1e293b;
    border-radius:14px;
    padding:1rem;
    margin-top:1rem;
}

/* TREATMENT */
.treatment-box {
    background:linear-gradient(135deg,rgba(14,165,233,.08),rgba(14,165,233,.02));
    border:1px solid rgba(14,165,233,.2);
    border-radius:14px;
    padding:1rem;
    margin-top:1rem;
}

/* PRECAUTIONS */
.prec-card {
    background:linear-gradient(135deg,rgba(52,211,153,.08),rgba(52,211,153,.02));
    border:1px solid rgba(52,211,153,.2);
    border-radius:12px;
    padding:0.9rem;
    margin-bottom:0.6rem;
}

/* WARN */
.warn-box {
    border-radius:12px;
}

/* LOW CONF */
.low-conf {
    border-radius:16px;
}

/* FOOTER */
.footer {
    color:#334155;
    opacity:.6;
    margin-top:2rem;
}

</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

# ── Curated treatment for all 41 diseases ─────────────────────────────────────
# MedQuad only covers ~18/41 diseases. This dict fills every gap.
TREATMENT_FALLBACK = {
    "malaria": "Treatment involves antimalarial medications such as artemisinin-based combination therapies (ACTs) or chloroquine depending on the Plasmodium species. Seek immediate medical care. Rest, hydration, and paracetamol for fever are supportive measures. Never self-medicate.",
    "dengue": "No specific antiviral exists. Treatment is supportive: rest, oral rehydration, and paracetamol for fever — avoid aspirin and ibuprofen as they increase bleeding risk. Monitor for warning signs (severe abdominal pain, bleeding, rapid breathing) and seek emergency care immediately.",
    "chicken pox": "Treatment is mainly supportive: calamine lotion for itching, paracetamol for fever, and keeping nails trimmed to prevent scratching. Antiviral acyclovir may be prescribed for high-risk patients. Keep the patient isolated to prevent spread.",
    "drug reaction": "Stop the suspected medication immediately and consult a doctor or go to the nearest hospital. For mild reactions, antihistamines or corticosteroids may be prescribed. Severe reactions (anaphylaxis) require emergency epinephrine. Always inform your doctor of all drug allergies.",
    "allergy": "Treatment includes antihistamines, decongestants, or corticosteroids depending on severity. Severe allergic reactions (anaphylaxis) require emergency epinephrine. Long-term relief can come from allergen immunotherapy prescribed by an allergist. Avoid known triggers.",
    "acne": "Topical retinoids, benzoyl peroxide, or salicylic acid for mild cases. Moderate-to-severe acne may require oral antibiotics, hormonal therapy, or isotretinoin prescribed by a dermatologist. Keep skin clean and avoid squeezing pimples.",
    "gerd": "Lifestyle changes: avoid trigger foods (spicy, fatty, acidic), eat smaller meals, and do not lie down after eating. Medications include antacids, H2 blockers (famotidine), or proton pump inhibitors (omeprazole). Consult a doctor if symptoms persist beyond 2 weeks.",
    "bronchial asthma": "Reliever inhalers (salbutamol) for immediate symptom relief; preventer inhalers (corticosteroids) for long-term control. Avoid known triggers. In severe attacks seek emergency care immediately. Every patient should have a written asthma action plan from their doctor.",
    "typhoid": "Treated with antibiotics prescribed by a doctor — commonly ciprofloxacin, azithromycin, or ceftriaxone. Rest and adequate fluid intake are essential. Complete the full antibiotic course even if you feel better to prevent relapse.",
    "vertigo paroymsal positional vertigo": "The Epley maneuver — a series of head-position changes performed by a physiotherapist or doctor — is the primary treatment and resolves most BPPV cases in one to three sessions. Vestibular rehabilitation exercises help prevent recurrence. Meclizine may relieve acute symptoms.",
    "alcoholic hepatitis": "Abstinence from alcohol is the single most critical step. Severe cases may be treated with corticosteroids (prednisolone) under specialist care. Nutritional support and management of complications (infection, bleeding) are essential.",
    "chronic cholestasis": "Ursodeoxycholic acid (UDCA) is commonly prescribed to improve bile flow. Itching is managed with cholestyramine or rifampicin. Fat-soluble vitamin supplementation (A, D, E, K) is important. Monitor liver function regularly.",
    "jaundice": "Treatment targets the underlying cause: antivirals or rest for viral hepatitis; surgery or ERCP for bile duct obstruction. Avoid alcohol and hepatotoxic drugs. Regular liver function tests are required.",
    "hepatitis a": "Usually resolves on its own with rest, adequate nutrition, and hydration. Avoid alcohol and hepatotoxic medications. No specific antiviral is needed for most patients. Prevention through vaccination is highly effective.",
    "hepatitis b": "Acute hepatitis B typically resolves without treatment. Chronic hepatitis B is treated with antivirals (tenofovir, entecavir). Regular monitoring of liver function and viral load is essential. Hepatitis B vaccination prevents infection.",
    "hepatitis c": "Direct-acting antivirals (DAAs) achieve cure rates above 95% in 8–12 weeks. Treatment is guided by virus genotype. Early treatment prevents cirrhosis and liver cancer. No vaccine exists — prevention relies on avoiding exposure.",
    "hepatitis d": "Only occurs alongside hepatitis B. Pegylated interferon-alpha is the main treatment. Hepatitis B vaccination indirectly prevents hepatitis D. Liver transplant may be required in severe cases.",
    "hepatitis e": "Usually self-limiting — resolves with rest and adequate hydration. Avoid alcohol. Ribavirin may be used in immunocompromised patients. Ensure safe drinking water and food hygiene to prevent reinfection.",
    "fungal infection": "Topical antifungals (clotrimazole, miconazole) for skin infections; oral antifungals (fluconazole, itraconazole) for extensive or systemic infections. Keep affected areas clean and dry. Complete the full treatment course to prevent recurrence.",
    "impetigo": "Topical antibiotic (mupirocin) for mild cases; oral antibiotics (cephalexin, amoxicillin-clavulanate) for widespread infection. Keep the area clean, avoid touching sores, and wash hands frequently.",
    "common cold": "No cure exists. Supportive treatment: rest, stay hydrated, saline nasal rinse, and paracetamol or ibuprofen for fever and aches. Antibiotics are not effective and should not be used.",
    "gastroenteritis": "Oral rehydration with ORS solution is the most important step. Rest and bland foods. Avoid dairy and fatty foods. Antibiotics only if a bacterial cause is confirmed by a doctor. Seek care immediately if dehydration is severe.",
    "heart attack": "Emergency — call an ambulance immediately. While waiting: chew aspirin 300 mg if not allergic. Hospital treatment includes reperfusion therapy (primary PCI or thrombolytics). Long-term: antiplatelet drugs, statins, beta-blockers, ACE inhibitors under cardiologist supervision.",
    "varicose veins": "Compression stockings, regular exercise, and elevating legs for mild cases. Medical procedures (sclerotherapy, laser ablation, or surgery) for more severe cases. Consult a vascular specialist.",
    "dimorphic hemmorhoids piles": "High-fibre diet, increased water intake, stool softeners, and sitz baths for mild cases. Topical creams (hydrocortisone) relieve symptoms. Procedures such as rubber band ligation, sclerotherapy, or haemorrhoidectomy for severe cases.",
    "cervical spondylosis": "Physiotherapy and targeted neck exercises are the main treatment. Pain relief with NSAIDs or paracetamol. A cervical collar may be used short-term. Severe nerve compression may require corticosteroid injections or surgical decompression.",
    "paralysis brain hemorrhage": "Emergency treatment aims to control bleeding and reduce intracranial pressure — requires immediate hospitalisation. Rehabilitation (physiotherapy, occupational therapy, speech therapy) is essential for recovery.",
    "urinary tract infection": "Antibiotics based on urine culture results — commonly trimethoprim, nitrofurantoin, or ciprofloxacin. Drink plenty of water. Complete the full antibiotic course. Recurrent UTIs may require low-dose prophylactic antibiotics.",
    "migraine": "Acute: triptans (sumatriptan), NSAIDs, paracetamol, or antiemetics. Preventive therapy for frequent migraines: beta-blockers, topiramate, amitriptyline, or CGRP inhibitors. Identify and avoid personal triggers. Rest in a quiet, dark room during attacks.",
    "hypertension": "Lifestyle: low-sodium diet, regular exercise, weight loss, limit alcohol. Medications: ACE inhibitors, ARBs, calcium channel blockers, diuretics, or beta-blockers. Regular blood pressure monitoring and follow-up are essential.",
    "diabetes": "Type 1: insulin therapy. Type 2: lifestyle changes first, then metformin, progressing to other agents or insulin as needed. Monitor blood glucose regularly. Regular HbA1c testing every 3–6 months.",
    "tuberculosis": "Six-month course of multiple antibiotics — isoniazid, rifampicin, pyrazinamide, ethambutol. Never skip doses; incomplete treatment causes drug resistance. Directly Observed Therapy (DOT) is recommended.",
    "pneumonia": "Bacterial pneumonia is treated with antibiotics (amoxicillin, azithromycin). Rest and increased fluid intake are important. Hospitalisation and oxygen therapy for severe cases. Pneumococcal and influenza vaccines help prevent pneumonia.",
    "hypothyroidism": "Levothyroxine (synthetic T4) taken daily on an empty stomach. Dose is adjusted based on TSH levels checked every 6–12 months. Most patients require lifelong treatment.",
    "hyperthyroidism": "Options: antithyroid drugs (methimazole, propylthiouracil), radioactive iodine therapy, or surgery. Beta-blockers manage symptoms while awaiting definitive treatment. Under endocrinologist supervision.",
    "hypoglycemia": "Immediate: consume 15–20 g fast-acting carbohydrates (glucose tablets, fruit juice, regular soda). Recheck blood sugar after 15 minutes. If unconscious, glucagon injection or IV dextrose is required.",
    "psoriasis": "Topical treatments (corticosteroids, vitamin D analogues, coal tar) for mild cases. Phototherapy (UVB) for moderate cases. Systemic medications (methotrexate, cyclosporine) or biologics for severe cases. Regular moisturising helps manage dry skin.",
    "osteoarthristis": "Exercise and physiotherapy. Weight management. Pain relief with paracetamol or topical/oral NSAIDs. Corticosteroid injections or joint replacement surgery for severe cases.",
    "arthritis": "Rheumatoid: NSAIDs for pain, DMARDs (methotrexate) to slow disease, biologics for severe cases. Osteoarthritis: exercise and pain relief. Gout: allopurinol to lower uric acid. Physiotherapy benefits all types.",
    "aids": "Antiretroviral therapy (ART) suppresses HIV to undetectable levels. Early treatment prevents progression to AIDS. Regular CD4 count and viral load monitoring. With ART, people with HIV can live long, healthy lives.",
    "peptic ulcer diseae": "H. pylori-caused: triple therapy (two antibiotics + proton pump inhibitor) for 7–14 days. NSAID-caused: stop the NSAID and use a proton pump inhibitor. Avoid smoking and alcohol. Follow up to confirm H. pylori eradication.",
}


# ── Core text cleaner (must be defined before loaders use it) ─────────────────
def _clean(t):
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(BASE, "symptoms_to_disease_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "symptoms_to_disease_model.pkl not found. "
            "Run the notebook to generate it and place it in the same folder as app.py."
        )
    return joblib.load(path)

@st.cache_resource
def load_model_lr():
    path = os.path.join(BASE, "model_lr.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_model_cnb():
    path = os.path.join(BASE, "model_cnb.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_precautions():
    path = os.path.join(BASE, "precautions_map.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_descriptions():
    path = os.path.join(BASE, "symptom_Description.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["Disease"].apply(_clean), df["Description"]))

@st.cache_data
def load_medquad_treatment():
    """Load MedQuad treatment entries — tries pkl first, falls back to CSV."""
    pkl_path = os.path.join(BASE, "medquad_df.pkl")
    csv_path = os.path.join(BASE, "medquad.csv")

    df = None
    if os.path.exists(pkl_path):
        df = joblib.load(pkl_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

    if df is None or "question" not in df.columns or "answer" not in df.columns:
        return {}

    df = df.copy()
    df["q_low"] = df["question"].str.lower().fillna("")
    df["f_low"] = df["focus_area"].str.lower().fillna("") if "focus_area" in df.columns else ""
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()

    treat_df = df[df["q_low"].str.contains(
        r"treatment|how.*treat|therap|manag", regex=True, na=False
    )]

    result = {}
    for _, row in treat_df.iterrows():
        focus = str(row.get("f_low", "")).strip()
        if focus and row["answer"]:
            text = row["answer"]
            if len(text) > 600:
                cut = text[:600]
                last = cut.rfind(".")
                text = cut[:last + 1] if last > 300 else cut.rstrip() + "..."
            if focus not in result:
                result[focus] = text
    return result

@st.cache_data
def load_symptoms_list():
    path = os.path.join(BASE, "DiseaseAndSymptoms.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    syms = set()
    for col in [c for c in df.columns if "Symptom" in c]:
        syms.update(df[col].dropna().str.strip().tolist())
    return sorted(syms)


# ── Load everything ───────────────────────────────────────────────────────────
try:
    model        = load_model()
    model_lr     = load_model_lr()
    model_cnb    = load_model_cnb()
    prec_map     = load_precautions()
    desc_map     = load_descriptions()
    mq_map       = load_medquad_treatment()
    symptom_list = load_symptoms_list()
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error(
        "❌ The loaded model does not support probability estimates. "
        "Make sure symptoms_to_disease_model.pkl is in the app folder — "
        "not best_model.pkl or best_text_model.pkl."
    )
    st.stop()

# Ensemble flag: True only when all 3 models loaded
_use_ensemble = (model_lr is not None) and (model_cnb is not None)


# ── Symptom normalizer ────────────────────────────────────────────────────────
# Maps natural language phrases → exact model symptom names.
# Works regardless of case, commas, or how the user phrases things.
SYMPTOM_MAP = {
    # fever
    "high fever": "high_fever", "fever": "high_fever", "very high fever": "high_fever",
    "mild fever": "mild_fever", "slight fever": "mild_fever", "low grade fever": "mild_fever",
    # pain
    "chest pain": "chest_pain", "chest ache": "chest_pain", "chest discomfort": "chest_pain",
    "stomach pain": "stomach_pain", "stomach ache": "stomach_pain", "tummy pain": "stomach_pain",
    "abdominal pain": "abdominal_pain", "belly pain": "belly_pain",
    "back pain": "back_pain", "lower back pain": "back_pain",
    "joint pain": "joint_pain", "joint ache": "joint_pain",
    "muscle pain": "muscle_pain", "body ache": "muscle_pain", "body pain": "muscle_pain",
    "neck pain": "neck_pain", "knee pain": "knee_pain", "hip pain": "hip_joint_pain",
    "pain behind the eyes": "pain_behind_the_eyes", "pain behind eyes": "pain_behind_the_eyes",
    "anal pain": "pain_in_anal_region",
    # headache
    "headache": "headache", "head ache": "headache", "head pain": "headache",
    "migraine": "headache",
    # cough / breathing
    "cough": "cough", "dry cough": "cough", "wet cough": "cough", "coughing": "cough",
    "breathlessness": "breathlessness", "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness", "cant breathe": "breathlessness",
    "runny nose": "runny_nose", "runny": "runny_nose", "blocked nose": "congestion",
    "congestion": "congestion", "nasal congestion": "congestion",
    "sore throat": "throat_irritation", "throat pain": "throat_irritation",
    "throat irritation": "throat_irritation", "phlegm": "phlegm", "mucus": "phlegm",
    "sneezing": "continuous_sneezing",
    # skin
    "itching": "itching", "itchy": "itching", "itch": "itching",
    "skin rash": "skin_rash", "rash": "skin_rash", "rashes": "skin_rash",
    "blister": "blister", "blisters": "blister",
    "skin peeling": "skin_peeling", "peeling skin": "skin_peeling",
    "yellowish skin": "yellowish_skin", "yellow skin": "yellowish_skin",
    "pus": "pus_filled_pimples", "pimples": "pus_filled_pimples",
    "blackheads": "blackheads",
    # eyes
    "red eyes": "redness_of_eyes", "pink eye": "redness_of_eyes",
    "yellow eyes": "yellowing_of_eyes", "yellowing of eyes": "yellowing_of_eyes",
    "watery eyes": "watering_from_eyes", "teary eyes": "watering_from_eyes",
    "blurred vision": "blurred_and_distorted_vision", "blurry vision": "blurred_and_distorted_vision",
    "visual disturbances": "visual_disturbances",
    # digestive
    "vomiting": "vomiting", "vomit": "vomiting", "throwing up": "vomiting", "nausea": "nausea",
    "diarrhoea": "diarrhoea", "diarrhea": "diarrhoea", "loose stool": "diarrhoea",
    "constipation": "constipation", "indigestion": "indigestion", "acidity": "acidity",
    "heartburn": "acidity", "loss of appetite": "loss_of_appetite", "no appetite": "loss_of_appetite",
    "excessive hunger": "excessive_hunger", "always hungry": "excessive_hunger",
    "bloating": "distention_of_abdomen", "bloated": "distention_of_abdomen",
    "gas": "passage_of_gases", "flatulence": "passage_of_gases",
    "stomach bleeding": "stomach_bleeding", "bloody stool": "bloody_stool",
    # energy / general
    "fatigue": "fatigue", "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue",
    "weakness": "fatigue", "weak": "fatigue",
    "lethargy": "lethargy", "lethargic": "lethargy", "no energy": "lethargy",
    "malaise": "malaise", "unwell": "malaise", "feeling unwell": "malaise",
    "weight loss": "weight_loss", "losing weight": "weight_loss",
    "weight gain": "weight_gain", "gaining weight": "weight_gain",
    # sweating / chills
    "sweating": "sweating", "sweat": "sweating", "night sweats": "sweating",
    "chills": "chills", "shivering": "shivering", "shiver": "shivering",
    "cold hands": "cold_hands_and_feets", "cold feet": "cold_hands_and_feets",
    # heart
    "fast heart rate": "fast_heart_rate", "palpitations": "palpitations",
    "heart pounding": "palpitations", "racing heart": "fast_heart_rate",
    # urine
    "dark urine": "dark_urine", "yellow urine": "yellow_urine",
    "frequent urination": "polyuria", "polyuria": "polyuria",
    "burning urination": "burning_micturition", "pain urinating": "burning_micturition",
    # swelling / joints
    "swollen legs": "swollen_legs", "swelling": "swelling_joints",
    "swollen joints": "swelling_joints", "stiff neck": "stiff_neck",
    "stiffness": "movement_stiffness", "joint stiffness": "movement_stiffness",
    # neuro / mood
    "dizziness": "dizziness", "dizzy": "dizziness", "lightheaded": "dizziness",
    "anxiety": "anxiety", "anxious": "anxiety", "depression": "depression", "depressed": "depression",
    "mood swings": "mood_swings", "irritability": "irritability", "irritable": "irritability",
    "loss of balance": "loss_of_balance", "unsteady": "unsteadiness",
    "slurred speech": "slurred_speech", "spinning": "spinning_movements",
    "lack of concentration": "lack_of_concentration", "cant focus": "lack_of_concentration",
    # other
    "dehydration": "dehydration", "thirsty": "dehydration",
    "obesity": "obesity", "obese": "obesity", "overweight": "obesity",
    "swollen lymph nodes": "swelled_lymph_nodes", "lymph nodes": "swelled_lymph_nodes",
    "enlarged thyroid": "enlarged_thyroid", "thyroid": "enlarged_thyroid",
}

# Sort longest phrases first so "high fever" matches before "fever"
_SORTED_PHRASES = sorted(SYMPTOM_MAP.keys(), key=len, reverse=True)

def normalize_free_text(text: str) -> str:
    """
    Converts what the user types into model-compatible symptom tokens.
    - Case insensitive (FEVER = fever = Fever)
    - Commas optional — works with or without them
    - Maps common phrases to exact symptom names the model was trained on
    """
    text = text.lower().strip()
    # treat commas, semicolons, "and" all as separators
    text = re.sub(r"[,;]+", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    found = []
    remaining = text
    for phrase in _SORTED_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, remaining):
            found.append(SYMPTOM_MAP[phrase])
            remaining = re.sub(pattern, " ", remaining)

    # keep leftover words too (helps with symptoms not in the map)
    leftover = re.sub(r"\s+", " ", remaining).strip()
    if leftover:
        found.append(leftover)

    return " ".join(found) if found else text

def count_recognized(text: str) -> int:
    """How many known symptoms were recognized in the text."""
    text = text.lower()
    text = re.sub(r"[,;]+", " ", text)
    count = 0
    for phrase in _SORTED_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            count += 1
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    return count

# ── Core functions ────────────────────────────────────────────────────────────
def predict_topk(inp, k=5):
    """Ensemble soft-voting across all 3 models (or single model fallback)."""
    inp = normalize_free_text(inp)
    inp = _clean(inp)
    if not inp:
        return []

    if _use_ensemble:
        # Weighted soft-voting: LinearSVC=0.5, LogReg=0.3, CNB=0.2
        weights = (0.5, 0.3, 0.2)
        all_classes = model.classes_
        n = len(all_classes)
        class_idx = {c: i for i, c in enumerate(all_classes)}

        def get_proba(m):
            p = m.predict_proba([inp])[0]
            m_cls = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
            aligned = np.zeros(n)
            for j, cls in enumerate(m_cls):
                if cls in class_idx:
                    aligned[class_idx[cls]] = p[j]
            return aligned

        blended = (weights[0] * get_proba(model) +
                   weights[1] * get_proba(model_lr) +
                   weights[2] * get_proba(model_cnb))
        top_idx = np.argsort(blended)[::-1][:k]
        return [(all_classes[i], float(blended[i])) for i in top_idx]
    else:
        # Fallback: single model
        proba = model.predict_proba([inp])[0]
        top_idx = np.argsort(proba)[::-1][:k]
        return [(model.classes_[i], float(proba[i])) for i in top_idx]

def get_precautions(name):
    key = _clean(name)
    if key in prec_map:
        return prec_map[key]
    for k in prec_map:
        if k in key or key in k:
            return prec_map[k]
    return []

def get_description(name):
    key = _clean(name)
    if key in desc_map:
        return desc_map[key]
    for k in desc_map:
        if k in key or key in k:
            return desc_map[k]
    return None

def get_treatment(name):
    """Priority: curated fallback dict → MedQuad."""
    key = _clean(name)
    if key in TREATMENT_FALLBACK:
        return TREATMENT_FALLBACK[key]
    for k in TREATMENT_FALLBACK:
        if k in key or key in k:
            return TREATMENT_FALLBACK[k]
    if key in mq_map:
        return mq_map[key]
    for k in mq_map:
        if k in key or key in k:
            return mq_map[k]
    return None

def confidence_level(top5):
    """
    Returns: 'high', 'medium', 'low', or 'none'
    - high:   show full result, no warning
    - medium: show result WITH a caution banner (symptoms match multiple diseases)
    - low:    show top possibilities only, ask for more symptoms
    - none:   truly unrecognizable input
    """
    if not top5:
        return "none"
    t1 = top5[0][1]
    t2 = top5[1][1] if len(top5) > 1 else 0.0
    margin = t1 - t2
    if t1 >= 0.30:
        return "high"
    if t1 >= 0.12:
        return "medium"
    if t1 >= 0.05:
        return "low"
    return "none"


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><h1>🩺 MediGuide AI</h1>'
    '<p>Describe your symptoms and get AI-powered disease insights, '
    'treatment information &amp; precautions</p></div>',
    unsafe_allow_html=True
)

# ── Model status badge ───────────────────────────────────────
if _use_ensemble:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1rem">'
        '<span style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);'
        'border-radius:99px;padding:4px 16px;font-size:.78rem;color:#34d399">'
        '⚡ Ensemble Mode — 3 models active: LinearSVC + LogReg + NaiveBayes'
        '</span></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1rem">'
        '<span style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);'
        'border-radius:99px;padding:4px 16px;font-size:.78rem;color:#fbbf24">'
        '⚠️ Single model mode — run Section 16 in notebook to enable ensemble'
        '</span></div>',
        unsafe_allow_html=True
    )

col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-label">Enter Symptoms</div>', unsafe_allow_html=True)

    selected_syms = st.multiselect(
        "Choose from known symptoms",
        options=symptom_list,
        format_func=lambda x: x.replace("_", " ").title(),
        placeholder="Search symptoms..."
    )
    free_text = st.text_area(
        "Or describe in free text (English)",
        placeholder="e.g. high fever headache chills sweating (commas optional)",
        height=110
    )
    combined = " ".join(selected_syms) + " " + free_text

    c1, c2 = st.columns([2, 1])
    with c1:
        diagnose = st.button("🔍 Diagnose", use_container_width=True)
    with c2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    # Live symptom recognition feedback
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
                f'<span style="display:inline-block;background:rgba(52,211,153,.12);'
                f'border:1px solid rgba(52,211,153,.3);border-radius:99px;'
                f'padding:3px 12px;font-size:.78rem;color:#6ee7b7;margin:3px 2px;">'
                f'✓ {p}</span>'
                for p in recognized_pills
            )
            st.markdown(
                f'<div style="margin-top:.6rem">'
                f'<div style="font-size:.7rem;color:#475569;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:4px;">Recognized symptoms</div>'
                f'{pills_html}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="margin-top:.6rem;background:rgba(251,191,36,.06);'
                'border:1px solid rgba(251,191,36,.2);border-radius:10px;'
                'padding:10px 14px;font-size:.82rem;color:#fbbf24;">'
                '⚠️ No symptoms recognized yet. Try words like:<br>'
                '<span style="color:#fcd34d">fever, headache, cough, chest pain, nausea, '
                'fatigue, vomiting, chills, sweating, dizziness</span>'
                '</div>',
                unsafe_allow_html=True
            )

    if selected_syms:
        pills = "".join(
            f'<span class="sym-pill">{s.replace("_", " ")}</span>'
            for s in selected_syms
        )
        st.markdown(f'<div style="margin-top:.5rem">{pills}</div>', unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e293b;margin:1.5rem 0">',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="margin-top:.5rem">'
        '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1rem 1.2rem;margin-bottom:.8rem">'
        '<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;color:#38bdf8;font-weight:600;margin-bottom:.6rem">📋 How to get the best result</div>'
        '<div style="color:#64748b;font-size:.82rem;line-height:1.8">'
        '<b style="color:#94a3b8">Option 1 — Type it out (easiest):</b><br>'
        'Just describe your symptoms in plain English in the text box above.<br>'
        '<span style="color:#475569">Example: </span><span style="color:#7dd3fc">high fever headache chills sweating nausea</span>'
        '</div></div>'
        '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:1rem 1.2rem;margin-bottom:.8rem">'
        '<div style="color:#64748b;font-size:.82rem;line-height:1.8">'
        '<b style="color:#94a3b8">Option 2 — Use the dropdown (most precise):</b><br>'
        'Pick exact symptom names from the list above when you know the medical term.<br>'
        '<span style="color:#475569">Examples: <span style="color:#7dd3fc">high_fever, chest_pain, vomiting, dizziness</span></span>'
        '</div></div>'
        '<div style="background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.15);border-radius:14px;padding:1rem 1.2rem;margin-bottom:.8rem">'
        '<div style="font-size:.82rem;line-height:1.8">'
        '<b style="color:#f87171">&#9888;&#65039; Do not mix unrelated symptoms from both inputs:</b><br>'
        '<span style="color:#94a3b8">Picking <b style="color:#f87171">Blackheads + Anxiety</b> from the dropdown '
        'AND typing <b style="color:#f87171">high fever chest pain</b> below causes low confidence '
        'because those symptoms belong to different diseases — the model gets confused.<br><br>'
        '<b style="color:#fca5a5">Tip:</b> Only enter symptoms you are actually feeling right now, all from the same illness.</span>'
        '</div></div>'
        '<div style="color:#334155;font-size:.78rem;line-height:1.7;padding:0 .2rem">'
        '&#10003; Commas optional &nbsp;&bull;&nbsp; &#10003; Any case (FEVER = fever) &nbsp;&bull;&nbsp; &#10003; Green badge = recognized'
        '</div></div>',
        unsafe_allow_html=True
    )

with col_results:
    st.markdown('<div class="section-label">Diagnosis Results</div>', unsafe_allow_html=True)

    if diagnose:
        if not combined.strip():
            st.warning("⚠️ Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                results = predict_topk(combined, k=5)

            if not results:
                st.error("Could not process input.")
            else:
                level = confidence_level(results)
                top_disease, top_conf = results[0]

                if level == "none":
                    # Truly unrecognizable input
                    st.markdown(
                        '<div class="low-conf">'
                        '⚠️ <b>Could not identify a condition</b><br>'
                        'Try adding more specific symptoms, e.g. fever, cough, headache.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                elif level == "low":
                    # Some signal but not enough — show possibilities without full card
                    st.markdown(
                        '<div style="background:rgba(251,191,36,.06);border:1px solid '
                        'rgba(251,191,36,.2);border-radius:12px;padding:1.2rem;margin-bottom:.8rem">'
                        '<div style="color:#fbbf24;font-weight:600;margin-bottom:.5rem">'
                        '⚠️ Low confidence — add more symptoms for a better result</div>'
                        '<div style="color:#94a3b8;font-size:.85rem;margin-bottom:.8rem">'
                        'These conditions match your symptoms but the model is uncertain:</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    for disease, conf in results[:3]:
                        bw = min(conf * 100, 100)
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="disease-name" style="font-size:1.05rem">{disease.title()}</div>
                            <div class="bar-bg"><div class="bar" style="width:{bw:.0f}%;opacity:.5"></div></div>
                            <div style="font-size:.78rem;color:#64748b">{conf*100:.1f}% confidence</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    # medium or high — show full result card
                    precs     = get_precautions(top_disease)
                    desc      = get_description(top_disease)
                    treatment = get_treatment(top_disease)
                    bar_w     = min(top_conf * 100, 100)

                    # Medium confidence: show a softer caution banner
                    if level == "medium":
                        st.markdown(
                            '<div style="background:rgba(251,191,36,.06);border:1px solid '
                            'rgba(251,191,36,.15);border-radius:10px;padding:8px 14px;'
                            'margin-bottom:.8rem;color:#fbbf24;font-size:.82rem;">'
                            '⚠️ Moderate confidence — your symptoms match several conditions. '
                            'Consider adding more details or using the dropdown.'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    about_html = ""
                    if desc:
                        about_html = (
                            f'<div class="about-box">'
                            f'<div class="about-label">About this condition</div>'
                            f'{desc}'
                            f'</div>'
                        )

                    treat_html = ""
                    if treatment:
                        treat_html = (
                            f'<div class="treatment-box">'
                            f'<div class="treatment-label">💊 Treatment Information</div>'
                            f'{treatment}'
                            f'</div>'
                        )

                    st.markdown(f"""
                    <div class="result-card top">
                        <div class="rank">🏆 Top Diagnosis</div>
                        <div class="disease-name">{top_disease.title()}</div>
                        <div class="bar-bg">
                            <div class="bar" style="width:{bar_w:.0f}%"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-size:.78rem;color:#64748b">Confidence</span>
                            <span style="font-size:.78rem;color:#94a3b8;font-weight:500">
                                {top_conf * 100:.1f}%
                            </span>
                        </div>
                        {about_html}
                        {treat_html}
                    </div>""", unsafe_allow_html=True)

                    if precs:
                        st.markdown(
                            '<div class="section-label">Recommended Precautions</div>',
                            unsafe_allow_html=True
                        )
                        for i, p in enumerate(precs, 1):
                            st.markdown(
                                f'<div class="prec-card">'
                                f'<div class="prec-num">{i}</div>'
                                f'<div class="prec-text">{p}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    if len(results) > 1:
                        st.markdown(
                            '<div class="section-label">Other Possibilities</div>',
                            unsafe_allow_html=True
                        )
                        for disease, conf in results[1:4]:
                            bw = min(conf * 100, 100)
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="disease-name" style="font-size:1.05rem">
                                    {disease.title()}
                                </div>
                                <div class="bar-bg">
                                    <div class="bar" style="width:{bw:.0f}%;opacity:.6"></div>
                                </div>
                                <div style="display:flex;justify-content:space-between">
                                    <span style="font-size:.78rem;color:#64748b">Confidence</span>
                                    <span style="font-size:.78rem;color:#94a3b8;font-weight:500">
                                        {conf * 100:.1f}%
                                    </span>
                                </div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown(
                        '<div class="warn-box">'
                        '⚕️ AI-generated results. Always consult a qualified doctor.'
                        '</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.markdown(
            '<div style="text-align:center;padding:4rem 1rem;color:#1e293b">'
            '<div style="font-size:3rem">🩺</div>'
            '<div style="font-family:Syne,sans-serif;font-size:1.1rem;'
            'color:#334155;margin-top:1rem">'
            'Enter your symptoms and click Diagnose'
            '</div></div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="footer">MediGuide AI • For educational purposes only</div>',
    unsafe_allow_html=True
)