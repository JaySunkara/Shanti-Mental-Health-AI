import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
#from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

dataset = pd.read_csv("data.csv")

dataset = dataset.drop("Answer.f1.jealous.raw", axis='columns')
dataset = dataset.drop("Answer.f1.awkward.raw", axis='columns')

def data_clean(entry):
  # Lowercase the texts
  entry = entry.lower()

  # Cleaning punctuations in the text
  punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  entry = entry.translate(punc)

  # Removing numbers in the text
  entry = re.sub(r'\d+', '', entry)

  # Remove possible links
  entry = re.sub('https?://\S+|www\.\S+', '', entry)

  # Deleting newlines
  entry = re.sub('\n', '', entry)

  return entry

Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])

def data_process(entry):
  Processed_Text = list()
  Lemmatizer = WordNetLemmatizer()

  # Tokens of Words
  Tokens = nltk.word_tokenize(entry)

  # Removing Stopwords and Lemmatizing Words
  # To reduce noises in our dataset, also to keep it simple and still 
  # powerful, we will only omit the word `not` from the list of stopwords

  for word in Tokens:
    if word not in Stopwords:
      Processed_Text.append(Lemmatizer.lemmatize(word))

  return(" ".join(Processed_Text))

dataset["Answer"] = dataset["Answer"].apply(lambda Text: data_clean(Text))
dataset["Answer"] = dataset["Answer"].apply(lambda Text: data_process(Text))
