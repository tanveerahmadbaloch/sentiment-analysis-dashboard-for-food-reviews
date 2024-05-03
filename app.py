import streamlit as st           
import  pandas as pd                   # Data manipulation and analysis
import re    
import nltk               
from nltk.tokenize import  sent_tokenize
nltk.download('punkt') 
import openpyxl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import plotly.express as px



st.set_page_config("NLP Analysis", layout="wide")
# st.title("Sentiment Analysis WebApp")
# st.header('_Streamlit_ is :blue[cool] :sunglasses:')
#A streamlit app with two centered texts with different seizes

st.markdown("<h1 style='text-align: center; color: grey;'>Sentiment Analysis</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: black;'>Get sentiment analysis of any file </h2>", unsafe_allow_html=True)


# create a icon in streamlit web where user can upload excel file
# uploaded_file=st.file_uploader("Chooose an excel file", type=["xlsx","xls"])
# # Check if a file has been uploaded
# if uploaded_file is not None:
# 	# Read the uploaded file into a Pandas DataFrame
# 	df = pd.read_excel(uploaded_file)
 
 
# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file into a Pandas DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Drop null values
    df.dropna(inplace=True)
    
    # Apply text preprocessing to the 'english text' column
    def preprocess_text(text):
        if isinstance(text, str):  
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.lower()
            sentences = sent_tokenize(text)
            text = ' '.join(sentences)
        return text
    
    df['english text'] = df['english text'].apply(preprocess_text)
    
    # Display a success message
    st.write("Preprocessing Done!")
    
    # Load the sentiment analysis model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Define a function to perform sentiment analysis
    def sentiment_analysis(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        return scores[:, 1].item()  # Get the positive score
    
    # Show a loading icon while performing sentiment analysis
    with st.spinner('Performing Sentiment Analysis...'):
        start_time = time.time()
        # Apply the sentiment analysis function to the text column
        df["sentiment_score"] = df["english text"].apply(sentiment_analysis)
        # Convert the sentiment score to a label (positive or negative)
        df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "positive" if x > 0.5 else "negative")
        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Sentiment Analysis Done! (Took {execution_time:.2f} seconds)")
    st.write(df.head(7))
    
    # Create a table to display sentiment analysis results
    sentiment_table = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 
                                'Count': [df['sentiment_label'].value_counts()['positive'], 
                                          df['sentiment_label'].value_counts()['negative']]})
    
    
    # Create a pie chart to display sentiment analysis results
    fig_pie = px.pie(sentiment_table, values='Count', names='Sentiment', title='Sentiment Analysis (Pie Chart)')
    
    
    # Create a bar chart to display sentiment analysis results
    fig_bar = px.bar(sentiment_table, x='Sentiment', y='Count', title='Sentiment Analysis (Bar Chart)')

# Display the table and charts in a 2x2 grid
    st.write("Sentiment Analysis Results:")
    st.write(sentiment_table)
    st.write(fig_pie)
    st.write(fig_bar)

    
    

   
   
   