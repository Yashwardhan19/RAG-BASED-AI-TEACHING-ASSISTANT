import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CourseGPT",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3250;
    }

    /* Chat messages */
    .user-msg {
        background: #1e3a5f;
        border-left: 3px solid #4a9eff;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0eaff;
    }
    .bot-msg {
        background: #1a1d27;
        border-left: 3px solid #7c3aed;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0e0e0;
    }

    /* Source card */
    .source-card {
        background: #12151f;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #a0aec0;
    }
    .source-card strong { color: #4a9eff; }

    /* Title */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    /* Input box */
    .stTextInput input {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3250 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #4a9eff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton button:hover {
        opacity: 0.9;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Hide press enter to apply */
    .stTextInput div[data-baseweb="input"] + div { display: none !important; }
    small.st-emotion-cache-eczf16 { display: none !important; }
    [data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load embeddings (cached) ──────────────────────────────────
@st.cache_resource
def load_data():
    try:
        df = joblib.load("embeddings.joblib")
        return df, True
    except Exception as e:
        return None, False


# ── RAG functions ─────────────────────────────────────────────
def create_embedding(text_list):
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=30)
        return r.json()["embeddings"]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def inference(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        return r.json()
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    seconds = int(seconds)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"


def ask_question(query, df, top_results=5):  # top_results fixed at 5
    # Create embedding
    embedding = create_embedding([query])
    if embedding is None:
        return None, None

    question_embedding = embedding[0]

    # Cosine similarity
    similarities = cosine_similarity(
        np.vstack(df['embedding'].values), [question_embedding]
    ).flatten()

    max_indx = similarities.argsort()[::-1][:top_results]
    top_df = df.loc[max_indx]

    # Build prompt
    prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{top_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

    response = inference(prompt)
    if response is None:
        return None, None

    return response["response"], top_df


# ── Session state init ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ── Load data ─────────────────────────────────────────────────
df, loaded = load_data()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    # Video titles list
    if loaded:
        st.markdown("**Videos**")
        videos = df[["number", "title"]].drop_duplicates(subset="number")
        videos["number"] = pd.to_numeric(videos["number"], errors="coerce")
        videos = videos.sort_values("number")
        for _, row in videos.iterrows():
            st.markdown(f"""
<div style="padding: 6px 10px; margin: 4px 0; border-radius: 6px; 
background-color: #1a1d27; border-left: 2px solid #4a9eff; font-size: 12px;">
    <span style="color:#4a9eff; font-weight:600;">#{int(row['number'])}</span>
    <span style="color:#c0caf5; margin-left:6px;">{row['title']}</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.error("❌ embeddings.joblib not found")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()


# ── Main area ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 CourseGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask anything about the course - I will tell you exactly which video and timestamp to watch.</div>', unsafe_allow_html=True)

# Chat history display
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">🧑‍💻 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)



# ── Input area ──────────────────────────────────────────────
st.divider()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Ask a question",
        placeholder="e.g. Where is Flexbox taught? How do I center a div?",
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}"
    )

with col2:
    ask_btn = st.button("Ask", use_container_width=True)


# ── Handle question ──────────────────────────────────────────
if ask_btn and user_input.strip() and user_input.strip() != st.session_state.last_query:
    if not loaded:
        st.error("embeddings.joblib not found. Cannot answer questions.")
    else:
        st.session_state.last_query = user_input.strip()
        st.session_state.input_key += 1
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Searching course content..."):
            response, top_df = ask_question(user_input, df)

        if response:
            msg_index = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if top_df is not None:
                st.session_state.sources[msg_index] = top_df

        st.rerun()