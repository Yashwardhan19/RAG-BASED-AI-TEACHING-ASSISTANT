# 🎓 CourseGPT — AI Teaching Assistant

> Ask anything about your course — get the exact video and timestamp to watch.

CourseGPT is a RAG (Retrieval-Augmented Generation) based AI teaching assistant built for video courses. It understands your course content and guides you to the right video and timestamp for any question you ask.

---

## 📸 Demo

### Home Screen
![CourseGPT Home](public/Screenshot_2026-06-04_123812.png)

### Asking a Question
![CourseGPT Response](public/Screenshot_2026-06-04_124054.png)

---

## ✨ Features

- 🔍 **Semantic Search** — Finds the most relevant video chunks using cosine similarity
- 🤖 **AI-Powered Answers** — LLM generates human-friendly responses with video + timestamp references
- 📚 **Video Library Sidebar** — Browse all course videos at a glance
- 💬 **Chat Interface** — Clean, dark-themed chat UI built with Streamlit
- ⚡ **Local & Private** — Powered by Ollama (runs fully on your machine, no API needed)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | Ollama (`bge-m3`) |
| LLM | Ollama (`llama3.2`) |
| Similarity Search | Scikit-learn (Cosine Similarity) |
| Data Storage | Joblib (Pickle) |
| Language | Python |

---

## 📁 Project Structure

```
CourseGPT/
│
├── videos/                  # Raw video files
├── mp3s/                    # Converted audio files
├── jsons/                   # Transcribed subtitle chunks
│
├── video_to_mp3.py          # Step 1: Convert videos to mp3
├── mp3_to_json.py           # Step 2: Transcribe mp3 to json chunks
├── preprocess_json.py       # Step 3: Generate embeddings → embeddings.joblib
├── process_incoming.py      # CLI version of the RAG pipeline
├── app.py                   # Streamlit web app
│
├── embeddings.joblib        # Saved embeddings dataframe
├── .env                     # API keys (never commit this!)
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com) installed and running locally

### Step 1 — Clone & Install Dependencies

```bash
git clone https://github.com/Yashwardhan19/CourseGPT.git
cd CourseGPT
pip install streamlit pandas scikit-learn joblib requests
```

### Step 2 — Pull Ollama Models

```bash
ollama pull llama3.2
ollama pull bge-m3
```

### Step 3 — Prepare Your Course Videos

```bash
# Convert videos to mp3
python video_to_mp3.py

# Transcribe mp3 to json chunks
python mp3_to_json.py

# Generate embeddings
python preprocess_json.py
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 💡 How It Works

```
User Query
    ↓
Generate Query Embedding (Ollama bge-m3)
    ↓
Cosine Similarity with all stored embeddings
    ↓
Top 5 most relevant video chunks selected
    ↓
Prompt sent to LLaMA 3.2 (Ollama) with context
    ↓
Human-friendly response with Video # and Timestamp
```

---

## ⚠️ Important

- `embeddings.joblib` can be large — consider adding it to `.gitignore` and regenerating locally
- Make sure Ollama is running before starting the app (`ollama serve`)

---

## 👤 Author

**Yashwardhan** — [GitHub](https://github.com/Yashwardhan19)


