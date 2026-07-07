# RNN Practical (Many to One)
# SMS Spam Detection USING SIMPLE RNN (MANY TO ONE)

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configurations

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# Clean Text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Train Model

def train_model():
    print("Training Dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels into numbers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean SMS
    df["text"] = df["text"].apply(clean_text)

    # Tokenization
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    x = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    y = df["label"]

    print("X Shape:", x.shape)
    print("Y Shape:", y.shape)

    # Save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Build RNN Model
    model = Sequential()

    model.add(Embedding(input_dim=MAX_WORDS, output_dim=32, input_length=MAX_LEN))
    model.add(SimpleRNN(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train Model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    # Save Model
    model.save(MODEL)

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test)

    print("\nAccuracy:", accuracy)

    predictions = (model.predict(x_test) > 0.5).astype(int)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


# Predict SMS

def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    probability = model.predict(sequence, verbose=0)[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", probability


# Train model only if not available

if not os.path.exists(MODEL):
    train_model()


# Streamlit UI

st.title("SMS Spam Detector")
st.write("Many to One RNN Example")

message = st.text_area("Enter your message here")

if st.button("Predict"):

    prediction, probability = predict_sms(message)

    st.success(prediction)
    st.write("Confidence:", round(probability * 100, 2), "%")