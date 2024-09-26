from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for

from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

from dotenv import load_dotenv


import time


from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate


# Load the .env file
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
def get_context_data(self, **kwargs):
    context = super(CLASS_NAME, self).get_context_data(**kwargs)
    return context


openai.api_key =  os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # Setting your OpenAI model

gfiles = glob.glob("documents1/*") # Reading your document directory

# Define functions to extract amount and substance
def extract_amount_from_text(text):
    import re
    amounts = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)  # Regex to extract amounts
    if amounts:
        return float(amounts[0].replace(",", ""))  # Remove commas and convert to float
    return 0  # Return 0 if no amount is found

def extract_substance_from_text(text):
    # Simple keyword-based extraction for substances
    if "substance" in text.lower():
        return text.split("substance", 1)[1].split()[0]  # Extract the word after "substance"
    return "unknown"  # Return "unknown" if no substance is found

# Loop through documents to create embeddings and extract structured data
for g1 in range(len(gfiles)):  # Iterating through every document

    f = open(f"embs{g1}.csv", "w")  # Creating a CSV file for storing the embeddings
    f.write("combined,amount,substance,embedding\n")  # Adding columns for combined text, amount, substance, and embedding
    f.close()

    content = ""
    with open(f"{gfiles[g1]}", 'r', errors='ignore') as file:  # Reading the document contents
        content += file.read()
        content += "\n\n"

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
    texts = text_splitter.split_text(content)  # Splitting the document into chunks

    def get_embedding(text, model="text-embedding-ada-002"):  # Function to get embeddings
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    df = pandas.read_csv(f"embs{g1}.csv")  # Read the CSV file

    # Process each chunk, extract amount and substance, and calculate embeddings
    combined_data = []
    for text_chunk in texts:
        amount = extract_amount_from_text(text_chunk)  # Extract the amount
        substance = extract_substance_from_text(text_chunk)  # Extract the substance
        embedding = get_embedding(text_chunk)  # Get the embedding

        # Append extracted data to the list
        combined_data.append([text_chunk, amount, substance, embedding])

    # Convert combined data to a dataframe and write to CSV
    df = pandas.DataFrame(combined_data, columns=["combined", "amount", "substance", "embedding"])

    # Save the dataframe to CSV
    df.to_csv(f"embs{g1}.csv", index=False)

    # Read the CSV file again to ensure embeddings are formatted correctly
    df = pandas.read_csv(f"embs{g1}.csv")

    # Convert embeddings into a list of floats
    embs = []
    for r1 in range(len(df.embedding)):
        e1 = df.embedding[r1].split(",") 
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs  # Update the embedding column with the formatted embeddings
    df.to_csv(f"embs{g1}.csv", index=False)  # Write the final version of the CSV file
