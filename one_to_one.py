import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

MODEL = "iris_rnn.keras"
SCALER = "scaler.pkl"
ENCODER = "encoder.pkl"


def train_model():

    df = pd.read_csv("Iris (1).csv")

    X = df.iloc[:, 1:5].values
    y = df["Species"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    y = to_categorical(y)

    with open(SCALER, "wb") as f:
        pickle.dump(scaler, f)

    with open(ENCODER, "wb") as f:
        pickle.dump(encoder, f)

    # One timestep with four features
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Sequential()

    model.add(
        SimpleRNN(
            32,
            input_shape=(1, 4)
        )
    )

    model.add(Dense(16, activation="relu"))

    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.2
    )

    model.save(MODEL)

    loss, acc = model.evaluate(X_test, y_test)

    print("Accuracy:", acc)


def predict_flower(values):

    model = load_model(MODEL)

    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)

    with open(ENCODER, "rb") as f:
        encoder = pickle.load(f)

    values = np.array(values).reshape(1, -1)

    values = scaler.transform(values)

    values = values.reshape((1, 1, 4))

    pred = model.predict(values, verbose=0)

    pred = np.argmax(pred)

    return encoder.inverse_transform([pred])[0]


if not os.path.exists(MODEL):
    train_model()

st.title("One-to-One RNN - Iris Flower Classification")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):

    result = predict_flower([sl, sw, pl, pw])

    st.success(result)