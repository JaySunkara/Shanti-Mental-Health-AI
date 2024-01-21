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

# import os
# import collections
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import tensorflow as tf
# import tensorflow_hub as hub
# from datetime import datetime
# import bert
# from bert import run_classifier
# from bert import optimization
# from bert import tokenization
# from bert import modeling


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
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['Tweet'] = df['Tweet'].apply(clean_text)
df['Tweet'] = df['Tweet'].str.replace('\d+', '')

MAX_NB_WORDS = 10000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
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
  temp_list.insert(0, 0)
  temp_list.insert(0, 0)
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
from keras.layers import (Dense, LSTM, Embedding, SpatialDropout1D)
from keras.models import Sequential
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 7
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


new_complaint = ['I won the lottery today! I am extremely happy about my kids and their future.']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['bruh', 'bruh', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
print(pred, labels[np.argmax(pred)])








# labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
# id2label = {idx:label for idx, label in enumerate(labels)}
# label2id = {label:idx for idx, label in enumerate(labels)}

# from transformers import AutoTokenizer
# import numpy as np

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def preprocess_data(examples):
#   # take a batch of texts
#   text = examples["Tweet"]
#   # encode them
#   encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
#   # add labels
#   labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
#   # create numpy array of shape (batch_size, num_labels)
#   labels_matrix = np.zeros((len(text), len(labels)))
#   # fill numpy array
#   for idx, label in enumerate(labels):
#     labels_matrix[:, idx] = labels_batch[label]

#   encoding["labels"] = labels_matrix.tolist()
  
#   return encoding

# encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

# example = encoded_dataset['train'][0]

# tokenizer.decode(example['input_ids'])

# [id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]

# encoded_dataset.set_format("torch")

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
#                                                            problem_type="multi_label_classification", 
#                                                            num_labels=len(labels),
#                                                            id2label=id2label,
#                                                            label2id=label2id)

# batch_size = 8
# metric_name = "f1"

# from transformers import TrainingArguments, Trainer

# args = TrainingArguments(
#     f"bert-finetuned-sem_eval-english",
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
# )

# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# from transformers import EvalPrediction
# import torch
    
# def multi_label_metrics(predictions, labels, threshold=0.5):
#     # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(torch.Tensor(predictions))
#     # next, use threshold to turn them into integer predictions
#     y_pred = np.zeros(probs.shape)
#     y_pred[np.where(probs >= threshold)] = 1
#     # finally, compute metrics
#     y_true = labels
#     f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
#     roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
#     accuracy = accuracy_score(y_true, y_pred)
#     # return as dictionary
#     metrics = {'f1': f1_micro_average,
#                'roc_auc': roc_auc,
#                'accuracy': accuracy}
#     return metrics

# def compute_metrics(p: EvalPrediction):
#     preds = p.predictions[0] if isinstance(p.predictions, 
#             tuple) else p.predictions
#     result = multi_label_metrics(
#         predictions=preds, 
#         labels=p.label_ids)
#     return result

# outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()
# trainer.evaluate()

# text = "I'm happy I can finally train a model for multi-label classification"

# encoding = tokenizer(text, return_tensors="pt")
# encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

# outputs = trainer.model(**encoding)

# logits = outputs.logits

# # apply sigmoid + threshold
# sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(logits.squeeze().cpu())
# predictions = np.zeros(probs.shape)
# predictions[np.where(probs >= 0.5)] = 1
# # turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)
