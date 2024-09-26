import streamlit as st
import os

from dotenv import load_dotenv

import glob
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import re
from PIL import Image

# Load the .env file
load_dotenv()
# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to extract the amount from text using regex
def extract_amount_from_text(text):
    amounts = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    if amounts:
        return float(amounts[0].replace(",", ""))
    return 0

# Function to extract the substance from text
def extract_substance_from_text(text):
    if "substance" in text.lower():
        return text.split("substance", 1)[1].split()[0]
    return "unknown"

# Function to process user queries and return relevant output
def logic(question, embeddings_index):
    csv_file_path = f"embs{embeddings_index}.csv"
    df = pd.read_csv(csv_file_path)

    embs = []
    for r1 in range(len(df.embedding)):
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    product_embedding = get_embedding(question)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df2 = df.sort_values("similarity", ascending=False)

    comb = [df2["combined"].iloc[0]]
    docs = [Document(page_content=t) for t in comb]
    
    return docs, df2

# Updated logic to handle all documents and sum amounts
def process_all_documents(question):
    gfiles = glob.glob("documents1/*")
    total_amount = 0
    combined_responses = []
    
    for g1 in range(len(gfiles)):
        docs, df2 = logic(question, embeddings_index=g1)
        combined_responses.append(docs[0].page_content)
        
        if "how much" in question.lower() and "substance" in question.lower():
            substance = extract_substance_from_text(question)
            relevant_rows = df2[df2['substance'].str.contains(substance, case=False)]
            total_amount += relevant_rows['amount'].sum()

    if total_amount > 0:
        return f"The total amount spent on the specified substance is {total_amount} dollars."
    else:
        prompt_template = question + """
        {text}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        output = chain.run([Document(page_content="\n".join(combined_responses))])
        return output

# Function to set page config and custom CSS
def set_page_config():
    st.set_page_config(layout="wide", page_title="")
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
        }
        .logo-container {
            margin-right: 20px;
        }
        .logo-container img {
            max-width: 100px;
            max-height: 100px;
        }
        .title-container {
            flex-grow: 1;
        }
        .stTextInput>div>div>input {
            min-height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to load and display logo
def display_logo(logo_path):
    try:
        if logo_path.startswith(('http://', 'https://')):
            return logo_path
        else:
            return Image.open(logo_path)
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        return None

# Streamlit app
def main():
    set_page_config()

    # Logo path
    logo_path = "./templates/assets/Image_EEG.jpg"  # Replace with your logo path or URL
    
    # Display logo and title side by side
    col1, col2 = st.columns([1, 4])
    
    with col1:
        logo = display_logo(logo_path)
        if logo is not None:
            st.image(logo, width=160)
        else:
            st.write("Logo")
    
    with col2:
        st.title("Ari, Your Generative AI Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = process_all_documents(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)  # Runs the app on port 8001