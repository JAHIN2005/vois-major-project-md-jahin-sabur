import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# --- PAGE SETUP (MUST BE FIRST) ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

# --- TITLE ---
st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: This system uses **Random Forest** to detect Network attacks 
and **Groq AI** to explain detected packets.
""")

# --- CONFIG ---
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# --- SIDEBAR ---
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (starts with gsk_)", type="password"
)
st.sidebar.caption("Get key from https://console.groq.com/keys")

st.sidebar.header("2. Model Training")

# --- DATA LOADER ---
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, nrows=3000)  # REDUCED ROWS (IMPORTANT)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

# --- MODEL TRAINING ---
def train_model(df):
    features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Fwd Packet Length Max',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Flow Packets/s'
    ]
    target = 'Label'

    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None, 0, None, None, None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=10, max_depth=10, random_state=42
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc, features, X_test, y_test

# --- LOAD DATA ---
st.write("📂 Loading dataset...")
df = load_data(DATA_FILE)

if df is None:
    st.error(f"❌ Dataset '{DATA_FILE}' not found")
    st.stop()

st.sidebar.success(f"Dataset Loaded: {len(df)} rows")

# --- TRAIN BUTTON ---
if st.sidebar.button("Train Model Now"):
    with st.spinner("Training model..."):
        clf, accuracy, features, X_test, y_test = train_model(df)

        if clf is not None:
            st.session_state.model = clf
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.sidebar.success(f"Training Complete! Accuracy: {accuracy:.2%}")

# --- DASHBOARD ---
st.header("3. Threat Analysis Dashboard")

if "model" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simulation")
        st.info("Pick a random packet from test data.")

        if st.button("🎲 Capture Random Packet"):
            idx = np.random.randint(0, len(st.session_state.X_test))
            st.session_state.packet = st.session_state.X_test.iloc[idx]
            st.session_state.actual = st.session_state.y_test.iloc[idx]

    if "packet" in st.session_state:
        packet = st.session_state.packet

        with col1:
            st.write("**Packet Details**")
            st.dataframe(packet.to_frame(name="Value"))

        with col2:
            st.subheader("AI Detection Result")

            prediction = st.session_state.model.predict(
                packet.values.reshape(1, -1)
            )[0]

            if prediction == "BENIGN":
                st.success("STATUS: SAFE (BENIGN)")
            else:
                st.error(f"🚨 STATUS: ATTACK DETECTED ({prediction})")

            st.caption(f"Actual Label: {st.session_state.actual}")

            st.markdown("---")
            st.subheader("Ask AI Analyst (Groq)")

            if st.button("Generate Explanation"):
                if not groq_api_key:
                    st.warning("Please enter Groq API Key")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)
                        prompt = f"""
You are a cybersecurity analyst.

Prediction: {prediction}

Packet details:
{packet.to_string()}

Explain briefly in simple student language.
"""
                        with st.spinner("Groq analyzing..."):
                            res = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.6,
                            )
                            st.info(res.choices[0].message.content)
                    except Exception as e:
                        st.error(e)
else:
    st.info("Click 'Train Model Now' to begin.")
