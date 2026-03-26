from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI(title="Spam Detector API")
ps = PorterStemmer()

# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("⚠️  vectorizer.pkl or model.pkl not found. Using mock predictions.")


def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


class MessageRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(req: MessageRequest):
    if not req.message.strip():
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)

    transformed = transform_text(req.message)

    if model_loaded:
        vectorized = tfidf.transform([transformed])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        is_spam = bool(prediction == 1)
        confidence = float(max(proba)) * 100
    else:
        # Mock for demo when model files are absent
        spam_keywords = ["win", "free", "prize", "click", "urgent", "congratulations", "offer", "cash", "credit"]
        is_spam = any(kw in transformed for kw in spam_keywords)
        confidence = 87.4 if is_spam else 94.1

    return {
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "NOT SPAM",
        "confidence": round(confidence, 2),
        "processed_text": transformed,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded}

import webbrowser

if __name__ == "__main__":
    import uvicorn
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)