import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
loaded_model = pickle.load(open("nnshanti.sav", "rb"))
tokenizer = Tokenizer(num_words=10000)

def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=250)
    return padded_sequences

def predict_text(text):
    preprocessed_text = preprocess_input(text)
    prediction = model.predict(preprocessed_text)
    return prediction

new_complaint = ['I am very happy.']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=250)
pred = loaded_model.predict(padded)
labels = ['bruh', 'bruh', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
print(pred, labels[np.argmax(pred)])

sample_text = "I'm super happy."
prediction = predict_text(sample_text)
print("Prediction:", prediction)

sample_text = "I am super happy and I love my mom."
prediction = predict_text(sample_text)
print("Prediction:", prediction)