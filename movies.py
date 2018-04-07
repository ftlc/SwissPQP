import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


# Import Data
# Downloaded from http://www.nltk.org/nltk_data/
moviedir = 'data/movie_reviews/txt_sentoken/'
reviews = load_files(moviedir, shuffle=True)

movie_train, movie_test, y_train, y_test = train_test_split(
    reviews.data, reviews.target, test_size=0.20
)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(movie_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = PCA(n_components=1700)
X_train_array = X_train_tfidf.toarray()
pca.fit(X_train_array)

# Plot the Principal Components
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
