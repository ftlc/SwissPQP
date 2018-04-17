# Trying to train a model to distinguish movie industry resumes from movie reviews

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

resumedir = 'data/resumes_and_reviews'
resumes = load_files(resumedir, shuffle=True)

# Hide some data in the vault to avoid snooping
x_work, x_vault, y_work, y_vault = train_test_split(
    resumes.data, resumes.target, test_size=0.10
)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svd', TruncatedSVD(n_components=20)),
                     ('clf', RandomForestClassifier()),
])

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
