import pandas as pd
import cohere
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cohere_client = cohere.Client('XSZYWxjzrkZ1YZ9KhFwSygYzKS5NqpVqkyB9rLKl')


data = pd.read_csv("dataset1.csv")
data['combined'] = data['User Question'] + " " + data['Bot Response']

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined'])

def find_relevant_context(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    most_similar_idx = similarities.argmax()
    return data.iloc[most_similar_idx]['combined']

def generate_response(user_input):
    context = find_relevant_context(user_input)
    prompt = f"Context: {context}\n\nUser Question: {user_input}\nBot Response (in 100 words):"

    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100,
        stop_sequences=["User Question:"]
    )
    return response.generations[0].text.strip()

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(lines=10, placeholder="Response will appear here..."),
    title="Chatbot",
    description="Ask me anything!"
)


iface.launch(share="True")
