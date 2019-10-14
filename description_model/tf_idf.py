import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import csv, random
import dateutil.parser
from sklearn.metrics import precision_recall_fscore_support


# Load data
train_texts = []
train_targets = []
test_texts = []
test_targets = []
with open('../data/X_train_desc.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        train_texts.append(row['description'])
        train_targets.append(int(row['label']))

test_data = []
with open('../data/X_test_desc.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_texts.append(row['description'])
        test_targets.append(int(row['label']))


# Create bag of words vectorizer to use scipy.sparse matrices
# and avoid wasting memory storing the many non-zero entries in the
# bag.
count_vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
X_counts_train = count_vect.fit_transform(train_texts)

# Apply tf-idf
tfidf_transformer = TfidfTransformer()
X_tfidf_train = tfidf_transformer.fit_transform(X_counts_train)
print(X_tfidf_train.shape)

# Train classifier
#classifier = LogisticRegression(C=1.0, intercept_scaling=1.0).fit(X_tfidf_train, train_targets)
classifier = BernoulliNB().fit(X_tfidf_train, train_targets)
#classifier = MLPClassifier(verbose=True, max_iter=10, hidden_layer_sizes=(100,100)).fit(X_tfidf_train, train_targets)

# Evaluate classifier
X_counts_test = count_vect.transform(test_texts)
X_tfidf_test = tfidf_transformer.transform(X_counts_test)
predictions = classifier.predict(X_tfidf_test)
print("ACCURACY: {}".format(np.mean(predictions == test_targets)))

p,r,f,_ = precision_recall_fscore_support(test_targets, predictions, beta=2, average='micro')
print("PRECISION: {}\nRECALL: {}\nF1: {}".format(p, r, f))


# Test on user input
test_string = "size 11 nike blazer hi top trainers in zebra print"
counts = count_vect.transform([test_string])
tfidf = tfidf_transformer.transform(counts)
predictions = classifier.predict(tfidf)
print("\nPredictions: {}".format(predictions))
