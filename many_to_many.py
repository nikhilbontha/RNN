# RNN Practical (Many-to-Many)
# Named Entity Recognition Using Simple RNN

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = "ner_model.keras"
TOKENIZER = "tokenizer.pkl"
ENCODER = "label_encoder.pkl"

MAX_WORDS = 10000
MAX_LEN = 30


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model():

    print("Training Model...")

    df = pd.read_csv("ner.csv", encoding="latin1")

    df.fillna(method="ffill", inplace=True)

    sentences = []
    labels = []

    sentence = []
    tag = []

    current = None

    for _, row in df.iterrows():

        if current != row["Sentence #"]:

            if sentence:
                sentences.append(sentence)
                labels.append(tag)

            sentence = []
            tag = []
            current = row["Sentence #"]

        sentence.append(str(row["Word"]))
        tag.append(str(row["Tag"]))

    sentences.append(sentence)
    labels.append(tag)

    # ---------------- Tokenizer ----------------

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post"
    )

    # ---------------- Label Encoder ----------------

    encoder = LabelEncoder()

    all_tags = []

    for t in labels:
        all_tags.extend(t)

    encoder.fit(all_tags)

    num_tags = len(encoder.classes_)

    Y = []

    for t in labels:

        encoded = encoder.transform(t)

        Y.append(encoded)

    Y = pad_sequences(
        Y,
        maxlen=MAX_LEN,
        padding="post"
    )

    # ---------------- Save Tokenizer ----------------

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    with open(ENCODER, "wb") as f:
        pickle.dump(encoder, f)

    # ---------------- Split Dataset ----------------

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- Model ----------------

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=64
        )
    )

    model.add(
        SimpleRNN(
            128,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                num_tags,
                activation="softmax"
            )
        )
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        Y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    loss, accuracy = model.evaluate(X_test, Y_test)

    print("Test Accuracy:", accuracy)

    model.save(MODEL)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict(sentence):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    with open(ENCODER, "rb") as f:
        encoder = pickle.load(f)

    seq = tokenizer.texts_to_sequences([sentence.split()])

    seq = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    pred = model.predict(seq, verbose=0)

    pred = np.argmax(pred, axis=-1)[0]

    words = sentence.split()

    result = []

    for word, label in zip(words, pred):

        result.append(
            (
                word,
                encoder.inverse_transform([label])[0]
            )
        )

    return result


# --------------------------------------------------
# Train Once
# --------------------------------------------------

if not os.path.exists(MODEL):
    train_model()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Named Entity Recognition Using Simple RNN")

text = st.text_input("Enter a Sentence")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        result = predict(text)

        st.subheader("Prediction")

        for word, label in result:
            st.write(f"**{word}**  ➜  {label}")