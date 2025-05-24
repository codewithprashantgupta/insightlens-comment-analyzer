import os
import streamlit as st
from transformers import pipeline
os.environ["PYTHONPATH"] = ""


# sentiment_analyzer = pipeline("sentiment-analysis")
# intent_classifier = pipeline("zero-shot-classification")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

st.title("Semantic Comment Analyzer")
st.write("Paste a comment or upload a CSV file to analyze sentiment and intent.")

# Input
input_text = st.text_area("Enter your comment:")

uploaded_file = st.file_uploader("Or upload a CSV with a 'comment' column", type=["csv"])

# Analysis Type
analysis_type = st.selectbox("Choose Analysis Type", ["Sentiment", "Intent"])

# Analyze Button
if st.button("Analyze"):
    if input_text.strip():
        if analysis_type == "Sentiment":
            result = sentiment_analyzer(input_text)[0]
            st.success(f"Sentiment: {result['label']} (Confidence: {round(result['score'] * 100, 2)}%)")

        elif analysis_type == "Intent":
            labels = ["complaint", "feedback", "query", "support"]
            result = intent_classifier(input_text, labels)
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            st.success(f"Intent: {top_label} (Confidence: {round(top_score * 100, 2)}%)")

    elif uploaded_file is not None:
        st.info("Uploaded file detected — batch processing will be available soon.")
        # You’ll implement this in Saturday’s batch logic

    else:
        st.warning("Please enter text or upload a file.")

