import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
loaded_model = None
tokenizer = Tokenizer(num_words=10000)


def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=250)
    return padded_sequences

def predict_text(text):
    loaded_model = load_model('bruh.h5')
    preprocessed_text = preprocess_input(text)
    prediction = loaded_model.predict(preprocessed_text)
    loaded_model = None
    return prediction

# new_complaint = ['I am very happy.']
# seq = tokenizer.texts_to_sequences(new_complaint)
# padded = pad_sequences(seq, maxlen=250)
# pred = loaded_model.predict(padded)
# labels = ['bruh', 'bruh', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
# print(pred, labels[np.argmax(pred)])

sample_text = "I hate everything and I want to dies."
prediction = predict_text(sample_text)
print("Prediction:", prediction)

# sample_text = "I am super happy and I love my mom."
# prediction = predict_text(sample_text)
# print("Prediction:", prediction)