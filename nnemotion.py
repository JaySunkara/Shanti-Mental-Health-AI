import numpy as np
import pandas as pd 
import re
import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from datasets import load_dataset

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
example = dataset['train'][0]
df = pd.DataFrame(dataset['train'])

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(nltk.corpus.stopwords.words("english")) - set(["not"])

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
df['Tweet'] = df['Tweet'].apply(clean_text)
df['Tweet'] = df['Tweet'].str.replace('\d+', '')

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Tweet'].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df['Tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

df.iloc[:, 2:] = df.iloc[:, 2:].astype(int)
temp_column = []
for row in range(len(df)):
  temp_list = []
  for emo in range(len(df.iloc[row])):
    if(df.iloc[row][emo] == True):
      temp_list.append(1)
    if(df.iloc[row][emo] == False):
       temp_list.append(0)
  temp_column.append(temp_list)
df['Sentiments'] = temp_column

X = tokenizer.texts_to_sequences(df['Tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
temp_column = pd.DataFrame(temp_column)
Y = temp_column.values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

import tensorflow as tf
from tensorflow import keras
from keras.layers import (Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional)
from keras.models import Sequential
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(11, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 7
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)


# model.save_weights("model_weights.h5")

# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


new_complaint = ['The experience at the restaurant was dreadful. The service was abysmal; we waited over an hour just to get our orders taken, and when the food finally arrived, it was cold and tasteless. On top of that, the ambiance was terrible, with loud music blaring from the speakers, making it impossible to hold a conversation. Overall, it was a complete waste of time and money.']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded).tolist()[0]
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
sorted_emotions = sorted(zip(labels, pred), key=lambda x: x[1], reverse=True)
for key, value in sorted_emotions:
	print(f"{key}: {value}")