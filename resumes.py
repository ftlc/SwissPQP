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

from scipy.stats import randint as sp_randint
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

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

#svd = TruncatedSVD(n_components=20)
#svd.fit(X_train_tfidf)
#print("svd explained variance: {}".format(svd.explained_variance_.sum()))
#x_train_svd = svd.transform(X_train_tfidf)

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_tfidf_test = tfidf_transformer.transform(X_test_count)
#x_test_svd = svd.transform(X_tfidf_test)


params = {'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': None,
    'max_features': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 457}

params = {'bootstrap': True,
    'criterion': 'entropy',
    'max_depth': 3,
    'max_features': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 457}


clf = RandomForestClassifier(**params)
clf.fit(X_train_tfidf, y_train)
predicted = clf.predict(X_tfidf_test)


print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
