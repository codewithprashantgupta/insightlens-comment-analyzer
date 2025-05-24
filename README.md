# InsightLens – Comment Analyzer with Hugging Face 🤖

InsightLens is a Streamlit-based app that uses Hugging Face Transformers to analyze the **sentiment** and **intent** of user comments.

## 🔍 Features
- 3-way Sentiment Analysis: Positive, Neutral, Negative
- Zero-shot Intent Classification: Complaint, Feedback, Query, Support
- Clean, minimal UI with instant insights
- Powered by 🤗 `cardiffnlp/twitter-roberta-base-sentiment-latest` and `facebook/bart-large-mnli`

## ▶️ Demo
![Demo](https://github.com/user-attachments/assets/cb5ad3e9-d57c-4902-b2be-3213c731cd32)


## 🚀 Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/codewithprashantgupta/insightlens-comment-analyzer.git
cd insightlens-comment-analyzer
```

2. **Create and activate a virtual environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate or cd venv\Scripts >> . ./activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

## 🌐 Deployment Options

### 🔸 Deploy with Docker

1. **Build the Docker image**
```bash
docker build -t insightlens .
```

2. **Run the container**
```bash
docker run -p 8501:8501 insightlens
```

### 🔸 Deploy with Docker
1. Push your project to a public GitHub repo (done ✅)
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
3. Create a new Space → Choose Streamlit as SDK → Connect your repo
4. Deploy instantly — no setup needed!


### 🔸 Deploy to Streamlit Community Cloud

1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)  
2. Log in with GitHub and select your repository  
3. Click **Deploy**  
4. Your app is live!
