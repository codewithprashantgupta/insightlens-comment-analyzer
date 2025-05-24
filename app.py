import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import torch

# Page config
st.set_page_config(page_title="InsightLens", layout="centered")
st.title("📊 InsightLens: Comment Analyzer")
st.write("Get instant insights into the tone and intent of customer comments using AI.")

# Loading tokenizer & model for sentiment (3-way)
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

sentiment_labels = ['Negative', 'Neutral', 'Positive']
sent_tokenizer, sent_model = load_sentiment_model()

# load zero-shot intent classifier
@st.cache_resource
def load_intent_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # Use CPU
    )

intent_classifier = load_intent_model()

# Function to analyze sentiment
def analyze_sentiment(text):
    encoded_input = sent_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = sent_model(**encoded_input)
    scores = output[0][0].numpy()
    probs = softmax(scores)
    return list(zip(sentiment_labels, probs))

# Dropdown
analysis_type = st.selectbox("🔍 Choose analysis type", ["Sentiment", "Intent"])

# Text area 
# Clear logic BEFORE text area is created
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""


input_text = st.text_area(
    "✍️ Enter your comment:",
    value=st.session_state["input_text"],
    key="input_text"
)



if st.button("Analyze"):
    if input_text.strip():
        if analysis_type == "Sentiment":
            sentiment_scores = analyze_sentiment(input_text)
            top_label, top_conf = max(sentiment_scores, key=lambda x: x[1])
            st.success(f"Sentiment: {top_label} (Confidence: {round(top_conf * 100, 2)}%)")

            with st.expander("Show all scores"):
                for label, score in sentiment_scores:
                    st.write(f"{label}: {round(score * 100, 2)}%")

        elif analysis_type == "Intent":
            candidate_labels = ["complaint", "feedback", "query", "support"]
            result = intent_classifier(input_text, candidate_labels)
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            st.success(f"Intent: {top_label} (Confidence: {round(top_score * 100, 2)}%)")

    else:
        st.warning("Please enter a comment to analyze.")
