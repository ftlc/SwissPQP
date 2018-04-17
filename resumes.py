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

svd = TruncatedSVD(n_components=20)
svd.fit(X_train_tfidf)
print("svd explained variance: {}".format(svd.explained_variance_.sum()))
x_train_svd = svd.transform(X_train_tfidf)

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_tfidf_test = tfidf_transformer.transform(X_test_count)
x_test_svd = svd.transform(X_tfidf_test)

clf = RandomForestClassifier()

parameters_rand = {
    "n_estimators": sp_randint(300, 2000),
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=parameters_rand,
                                   n_iter=n_iter_search,
                                   n_jobs=-1)

random_search.fit(x_train_svd, y_train)
predicted = random_search.predict(x_test_svd)
print("Parameters: {}".format(random_search.best_params_))


print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
