from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import *
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space"])
x = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

parametrs = {'C': power(10.0, arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(
    clf,
    parametrs,
    scoring="accuracy",
    n_jobs=-1
)
gs.fit(x, y)
C_best = gs.best_params_.get('C')

best_clf = SVC(C=C_best, kernel="linear", random_state=241)

best_clf.fit(x, y)
