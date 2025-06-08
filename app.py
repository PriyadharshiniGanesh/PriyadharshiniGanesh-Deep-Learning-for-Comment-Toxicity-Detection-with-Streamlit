import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Title and description
st.title('Toxicity Comment Detection')
st.write('Enter a comment to check if it contains toxic content')

# Text input
user_input = st.text_area("Comment:", "", height=150)

# Load model and vectorizer (pretrained)
# In a real app, you would load your trained model like:
# model = joblib.load('toxicity_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Prediction function (mock version)
def predict_toxicity(comment):
    # This is simplified - real app would use actual model
    cleaned_comment = clean_text(comment)
    
    # Mock prediction logic based on keywords
    toxic_keywords = ['fuck', 'shit', 'kill', 'hate', 'stupid', 'idiot', 'bullshit',
                      'asshole', 'die', 'cunt', 'faggot', 'nigger', 'retard']
    
    toxic_count = sum(1 for word in toxic_keywords if word in cleaned_comment)
    
    # Simple heuristic: more keywords = higher toxicity probability
    toxicity_prob = min(0.99, toxic_count * 0.3)
    return toxicity_prob

if st.button('Analyze Comment'):
    if user_input.strip() == "":
        st.warning("Please enter a comment to analyze")
    else:
        with st.spinner('Analyzing...'):
            # Get prediction
            toxicity_prob = predict_toxicity(user_input)
            
            # Display results
            st.subheader("Analysis Result")
            
            if toxicity_prob == 0:
                st.subheader('**Not Toxic**')
            else:
                st.subheader('**Toxic**')
            st.text('')
            # Show explanation
            with st.expander("Analysis Details"):
                st.write("**Processed text:**", clean_text(user_input))
                st.write("**Toxicity score:**", f"{toxicity_prob:.2f} (0 = clean, 1 = toxic)")
                st.write("Note: This demo uses simplified heuristics. A production system would use a trained machine learning model.")

# Add footer
st.markdown("---")
st.caption("Note: This is a demonstration app. Real toxicity classification requires proper machine learning models trained on comprehensive datasets.")