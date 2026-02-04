import string
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =====================
# APP
# =====================
app = FastAPI(title="Spam Detection API v1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# LOAD ASSETS
# =====================
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# =====================
# PREPROCESS
# =====================
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# =====================
# SCHEMA
# =====================
class InputText(BaseModel):
    text: str

# =====================
# ROUTES
# =====================
@app.post("/analyze")
def analyze_message(data: InputText):
    processed = preprocess(data.text)
    vec = vectorizer.transform([processed])

    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index]

    label = "SPAM" if prediction == "spam" else "SAFE"

    return {
        "label": label,
        "confidence": round(confidence * 100, 2)
    }

@app.get("/health")
def health():
    return {"status": "Ready"}
