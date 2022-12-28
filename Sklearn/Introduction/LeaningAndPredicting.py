# load in the digits dataset
from sklearn import datasets
digits = datasets.load_digits()

# Create a classifier that implements support vector classification (this is treated as a black box)
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

# Pass all but the last item from the dataset and fit them within the model to learn from
clf.fit(digits.data[:-1], digits.target[:-1])

# Predict the using the last image no in the dataset
print(clf.predict(digits.data[-1:]))