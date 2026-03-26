from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Safe NLTK setup (Render friendly)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = FastAPI(title="Spam Detector API")
ps = PorterStemmer()

# ✅ Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    model_loaded = True
except:
    model_loaded = False
    print("⚠️ Model files not found, using mock mode")


# ✅ Text preprocessing
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


# ✅ Request schema
class MessageRequest(BaseModel):
    message: str


# ✅ Home route (HTML)
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ✅ Prediction route
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
        # fallback demo mode
        spam_keywords = ["win", "free", "prize", "click", "urgent", "offer", "cash"]
        is_spam = any(kw in transformed for kw in spam_keywords)
        confidence = 85.0 if is_spam else 95.0

    return {
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "NOT SPAM",
        "confidence": round(confidence, 2),
        "processed_text": transformed,
    }


# ✅ Health check
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded}