# 📱 WhatsApp Chat Analyzer & NLP Classifier

## 🚀 Project Overview
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://srinidhi-g25-nlp-chat-analyzer-app-ibhxi1.streamlit.app)

**🔗 Live Demo:** [Try it here](https://srinidhi-g25-nlp-chat-analyzer-app-ibhxi1.streamlit.app) — no setup needed, just type a message and see who the AI thinks sent it.

This project is an end-to-end Machine Learning pipeline that ingests raw, unstructured WhatsApp chat exports and builds a predictive NLP classification model. The engine analyzes group texting habits, extracts unique linguistic signatures, and predicts the sender of a given text message based on their mathematical vocabulary profile.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Wrangling:** Pandas, Regular Expressions (Regex)
* **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`)
* **Visualization:** Matplotlib, Seaborn

## 🧠 Key Engineering Highlights
1. **Dynamic Data Anonymization:** Engineered a parsing script that automatically identifies real names and converts them into safe, anonymized aliases (`Person_1`, `Person_2`, etc.) to ensure data privacy.
2. **PII Leakage Prevention:** Implemented a custom stop-word threshold to strip Personally Identifiable Information (PII) from the model's vocabulary, preventing the AI from memorizing sensitive data as predictive cheat codes.
3. **Advanced Feature Extraction (TF-IDF):** Transformed messy text data into a vectorized mathematical matrix, automatically filtering out common filler words while boosting the mathematical weight of highly unique slang and typing patterns.
4. **Handling Class Imbalance:** Deployed a Balanced Logistic Regression model to penalize majority-class guessing, forcing the algorithm to accurately learn the behavior of less active users.
5. **Real-Time Inference Engine:** Built a deployment-ready function that allows users to input raw text on the fly and receive an instant prediction alongside a mathematical confidence probability score.

## 📊 Visualizing the AI's Brain
The project includes a detailed **Confusion Matrix Heatmap**, visually mapping out the mathematical overlap between different users' texting styles, and a **Signature Vocabulary Extractor** that prints the top 10 most heavily weighted words for each user.

## 💻 How to Run the Project
> **Just want to try it?** Skip all this and use the [live demo](https://srinidhi-g25-nlp-chat-analyzer-app-ibhxi1.streamlit.app) instead.

1. Clone this repository.
2. Ensure you have `pandas`, `scikit-learn`, `matplotlib`, and `seaborn` installed in your Python environment.
3. Open `Chat_Classifier.ipynb` in a Jupyter Notebook environment.
4. Run the cells sequentially from top to bottom.
5. Play with the `guess_who_said_this()` function at the bottom to test the real-time inference!
