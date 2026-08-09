import streamlit as st
import joblib

# 1. Load the exported brains
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# 2. Build the App UI
st.title("📱 Guess Who? WhatsApp Chat AI")
st.write("Type a fake message below and the AI will predict which of my friends sent it based on our group chat's mathematical texting patterns!")

# 3. Create the input box and prediction logic
user_input = st.text_input("Enter a test message:")

if st.button("Predict Sender"):
    if user_input:
        # Transform the text into math
        vectorized_text = vectorizer.transform([user_input])
        
        # Get the AI's prediction and confidence score
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = max(probabilities) * 100
        
        # Display the results
        st.success(f"🤖 AI Prediction: {prediction}")
        st.info(f"📊 Confidence Score: {confidence:.2f}%")
    else:
        st.warning("Please enter a message first!")