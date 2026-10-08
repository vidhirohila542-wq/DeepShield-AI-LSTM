"""
DeepShield AI
Deep Learning project using an LSTM neural network.

0 = SAFE
1 = SCAM / SUSPICIOUS

Educational project. The included dataset is synthetic.
"""

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "messages_dataset.csv"
MAX_WORDS = 3000
MAX_LENGTH = 25
EPOCHS = 12
BATCH_SIZE = 8


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def build_model():
    model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LENGTH),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def predict_message(model, tokenizer, message):
    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")

    probability = float(model.predict(padded, verbose=0)[0][0])

    if probability >= 0.5:
        label = "SUSPICIOUS / SCAM"
        confidence = probability * 100
    else:
        label = "SAFE"
        confidence = (1 - probability) * 100

    print("\nMessage:", message)
    print("Prediction:", label)
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 55)


def main():
    print("=== DeepShield AI ===")

    data = pd.read_csv(DATA_FILE)
    data["message"] = data["message"].apply(clean_text)

    X = data["message"]
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=MAX_LENGTH,
        padding="post"
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=MAX_LENGTH,
        padding="post"
    )

    model = build_model()

    print("\nTraining neural network...")
    model.fit(
        X_train_seq,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    probabilities = model.predict(X_test_seq, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    print("\n=== Model Evaluation ===")
    print(f"Test Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Safe", "Scam"],
        zero_division=0
    ))

    model.save("deepshield_lstm.keras")
    print("Saved trained model as deepshield_lstm.keras")

    print("\n=== Sample Predictions ===")
    samples = [
        "Congratulations you won a prize click the link to claim now",
        "Please send me the notes from today's lecture",
        "Your bank account will be blocked verify your details immediately"
    ]

    for sample in samples:
        predict_message(model, tokenizer, sample)


if __name__ == "__main__":
    main()
