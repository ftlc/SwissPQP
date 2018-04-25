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

'''
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
'''

resumes = load_files('actingAndManagerResumes/', shuffle=True)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)
