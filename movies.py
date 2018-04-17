import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import Data
# Downloaded from http://www.nltk.org/nltk_data/
moviedir = 'data/movie_reviews/txt_sentoken/'
reviews = load_files(moviedir, shuffle=True)

# Hide some data in the vault to avoid snooping
x_work, x_vault, y_work, y_vault = train_test_split(
    reviews.data, reviews.target, test_size=0.10
)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = PCA(n_components=20)
X_train_array = X_train_tfidf.toarray()
pca.fit(X_train_array)

X_train_pca = pca.transform(X_train_array)


# Plot the Principal Components
plt.figure()
plt.plot(pca.explained_variance_.cumsum())
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_tfidf_test = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_tfidf_test.toarray())

clf = RandomForestClassifier()

# specify parameters and distributions to sample from
parameters_rand = {
    "n_estimators": sp_randint(300, 2000),
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

# run randomized search
# Accuracy should be comparable to grid search, but runs much much faster
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=parameters_rand,
                                   n_iter=n_iter_search,
                                   n_jobs=-1)

random_search.fit(X_train_pca, y_train)

predicted = random_search.predict(X_test_pca)

print(np.mean(predicted == y_test))


#  # use a full grid over all parameters
#  param_grid_grid = {
    #  "n_estimators": [30, 50],
    #  "max_depth": [3, None],
    #  "max_features": [1, 3, 10],
    #  "min_samples_split": [2, 3, 10],
    #  "min_samples_leaf": [1, 3, 10],
    #  "bootstrap": [True, False],
    #  "criterion": ["gini", "entropy"]
#  }

#  # run grid search
#  grid_search = GridSearchCV(clf, param_grid=param_grid_grid, n_jobs=-1)
#  grid_search.fit(X_train_pca, y_train)

#  predicted_grid = grid_search.predict(X_test_pca)
#  np.mean(predicted == y_test)
