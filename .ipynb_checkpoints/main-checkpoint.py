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

Columns = ["Answer.f1.jealous.raw", "Answer.f1.awkward.raw", "Answer.t1.exercise.raw", "Answer.t1.family.raw", "Answer.t1.food.raw", "Answer.t1.friends.raw", "Answer.t1.god.raw", "Answer.t1.health.raw", "Answer.t1.recreation.raw", "Answer.t1.school.raw", "Answer.t1.sleep.raw", "Answer.t1.work.raw"]

dataset = dataset.drop(columns = Columns, axis = 1)

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

dataset.iloc[:, 1:] = dataset.iloc[:, 1:].astype(int)

melted_df = pd.melt(dataset, id_vars=['Answer'], var_name=None, value_name="True") 
result_df = melted_df.groupby('Answer')['True'].agg(list)

X = result_df.index
Y = result_df.values



# y_train = ["afraid", "angry", "anxious","ashamed","bored","calm","confused","disgusted","excited","frustrated","happy", "nostalgic", "proud", "sad", "satisfied", "suprised", "love"]
# x_train = dataset.loc[:,"Answer"].values

# Encoder = LabelEncoder()
# dataset["Answer.f1.afraid.raw","Answer.f1.angry.raw","Answer.f1.anxious.raw","Answer.f1.ashamed.raw","Answer.f1.bored.raw","Answer.f1.calm.raw","Answer.f1.confused.raw","Answer.f1.disgusted.raw","Answer.f1.excited.raw","Answer.f1.frustrated.raw","Answer.f1.happy.raw","Answer.f1.jealous.raw","Answer.f1.nostalgic.raw","Answer.f1.proud.raw","Answer.f1.sad.raw","Answer.f1.satisfied.raw","Answer.f1.surprised.raw","Answer.t1.exercise.raw","Answer.t1.family.raw","Answer.t1.food.raw","Answer.t1.friends.raw","Answer.t1.god.raw","Answer.t1.health.raw","Answer.t1.love.raw","Answer.t1.recreation.raw","Answer.t1.school.raw","Answer.t1.sleep.raw","Answer.t1.work.raw"] = Encoder.fit_transform(dataset["Answer.f1.afraid.raw","Answer.f1.angry.raw","Answer.f1.anxious.raw","Answer.f1.ashamed.raw","Answer.f1.bored.raw","Answer.f1.calm.raw","Answer.f1.confused.raw","Answer.f1.disgusted.raw","Answer.f1.excited.raw","Answer.f1.frustrated.raw","Answer.f1.happy.raw","Answer.f1.jealous.raw","Answer.f1.nostalgic.raw","Answer.f1.proud.raw","Answer.f1.sad.raw","Answer.f1.satisfied.raw","Answer.f1.surprised.raw","Answer.t1.exercise.raw","Answer.t1.family.raw","Answer.t1.food.raw","Answer.t1.friends.raw","Answer.t1.god.raw","Answer.t1.health.raw","Answer.t1.love.raw","Answer.t1.recreation.raw","Answer.t1.school.raw","Answer.t1.sleep.raw","Answer.t1.work.raw"])

# dataset["sentiment"].value_counts()
TF_IDF = TfidfVectorizer(max_features = 5000, ngram_range = (2, 2)) # Fitting and transforming our reviews into a matrix of weighed words 
# This will be our independent features 
X = TF_IDF.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
