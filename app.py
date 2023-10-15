import streamlit as st
import nltk
import random
import numpy as np
import pickle
from keras.models import load_model
from datetime import datetime
import json
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

import time


nltk.download('punkt')

# Initialize your chatbot model and data
model = load_model("chatbot.pkl")
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']

# Load intents from data.json
with open('intents.json') as json_data:
    intents = json.load(json_data)

st.title("HariMindscape")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to clean up the sentence and return the bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    bag = np.array(bag)
    return bag

# Constants
ERROR_THRESHOLD = 0.30

# Function to classify the sentence
def classify(sentence):
    bag = bow(sentence, words)
    results = model.predict(np.array([bag]))
    results = [[i, r] for i, r in enumerate(results[0]) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [(classes[r[0]], r[1]) for r in results]
    return return_list

# Function to get a response from your chatbot
def get_response(sentence):
    results = classify(sentence)
    if results:
        for intent in intents['intents']:
            if intent['tag'] == results[0][0]:
                response_options = intent['responses']
                return random.choice(response_options)
    return "I'm sorry, I don't understand."

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("You:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get a response from your chatbot
    bot_response = get_response(user_input)

    # Display bot response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate the bot's response with a delay
        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
