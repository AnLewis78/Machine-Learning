# load in the digits dataset
from sklearn import datasets
from sklearn.linear_model import GammaRegressor
digits = datasets.load_digits

# Create a classifier that implements support vector classification (this is treated as a black box)
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

