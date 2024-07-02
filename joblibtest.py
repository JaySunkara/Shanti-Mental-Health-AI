import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def load_model():
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    model.load_weights("model_weights.h5")
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=250)
    return padded_sequences

def predict_text(text, model):
    preprocessed_text = preprocess_input(text)
    prediction = model.predict(preprocessed_text)
    return prediction.tolist()

def analyze():
    nn = load_model()

    sample_text = input("Enter text: ")
    prediction = predict_text(sample_text, nn)[0]
    labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    sorted_emotions = sorted(zip(labels, prediction), key=lambda x: x[1], reverse=True)
    for key, value in sorted_emotions:
        print(f"{key}: {value}")

analyze()