from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

question_templates = [
    'What is the return of {} ?',
    'When was this document made ?',
    'What is the document about ?',
    'What is the appraised value of {}'
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def recommend_using_embeddings(embeddings):
    # Your logic for processing embeddings and generating questions goes here
    # For now, just returning a sample message
    return "Questions generated based on document embeddings."

@app.route('/')
def home():
    return render_template("bot.html")  # Assuming you have an HTML file for your web app

@app.route('/recommend', methods=['POST'])
def recommend():
    folder_path = 'documents1'  # Default folder path
    embeddings_file_path = 'embs0.csv'  # Use the specified filename

    try:
        # Read embeddings from CSV file
        embeddings_df = pd.read_csv(embeddings_file_path)
        embeddings = embeddings_df['embedding'].tolist()

        if not embeddings:
            return jsonify({'error': 'No embeddings found in the CSV file.'}), 400

        result = recommend_using_embeddings(embeddings)
        return jsonify({'message': result})

    except FileNotFoundError:
        return jsonify({'error': 'CSV file not found.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=8001, debug=True)
