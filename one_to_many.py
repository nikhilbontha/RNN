# RNN Practical (One to Many)
# Text Generation Using Simple RNN

import os
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ================= CONFIGURATION =================

MODEL = "text_generator.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
SEQ_LEN = 3

# ================= TRAIN MODEL =================

def train_model():

    print("Training Model...")

    # Dataset
    lines = [
        "machine learning is amazing",
        "deep learning is powerful",
        "artificial intelligence is the future",
        "python is easy to learn",
        "streamlit makes deployment easy",
        "data science is interesting",
        "i love machine learning",
        "rnn is used for sequence data",
        "tensorflow is a deep learning library",
        "neural networks are powerful",
        "india is a beautiful country",
        "chatgpt helps students learn",
        "simple rnn is easy to understand",
        "artificial intelligence changes the world",
        "deep learning uses neural networks"
    ]

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(lines)

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in lines:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(SEQ_LEN, len(token_list)):
            sequence = token_list[i-SEQ_LEN:i+1]
            input_sequences.append(sequence)

    input_sequences = np.array(input_sequences)

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    y = to_categorical(y, num_classes=total_words)

    model = Sequential()

    model.add(Embedding(total_words, 32, input_length=SEQ_LEN))
    model.add(SimpleRNN(128))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X,
        y,
        epochs=300,
        batch_size=8,
        verbose=1
    )

    model.save(MODEL)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    print("Model Saved Successfully!")

# ================= GENERATE TEXT =================

def generate_text(seed_text, next_words=8):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=SEQ_LEN,
            padding="pre"
        )

        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted:
                output_word = word
                break

        if output_word == "":
            break

        seed_text += " " + output_word

    return seed_text

# ================= TRAIN IF MODEL DOESN'T EXIST =================

if not os.path.exists(MODEL):
    train_model()

# ================= STREAMLIT UI =================

st.title("One to Many RNN")
st.subheader("Text Generation using Simple RNN")

text = st.text_input("Enter starting word(s):")

if st.button("Generate"):

    if text.strip() == "":
        st.warning("Please enter a word.")
    else:
        result = generate_text(text)
        st.success(result)