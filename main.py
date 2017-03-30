
import numpy as np
from scipy import special
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification

def lossf(w, X, y, l1, l2):
    lossf = float(sum(special.log1p(np.exp(-y.reshape(len(y), 1)*np.dot(X,w.reshape(len(w), 1))))) + l1 * sum(abs(w)) + l2 * sum(w**2))
    return lossf

def gradf(w, X, y, l1, l2):
    gradw = np.squeeze(np.dot(-X.T * y, 1 - 1 / (1 + np.exp(-y.reshape(len(y), 1)*np.dot(X,w.reshape(len(w), 1))))).T + l1 * np.sign(w) + l2 * 2 * w )
    return gradw

class LR(ClassifierMixin, BaseEstimator):
    def __init__(self, lr=1, l1=1e-4, l2=1e-4, num_iter=1000, verbose=0):
        self.l1 = l1
        self.l2 = l2
        self.w = None
        self.lr = lr
        self.verbose = verbose
        self.num_iter = num_iter

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.ones(d)/10
        for i in range(self.num_iter):
            self.w -= self.lr * gradf(self.w, X, y, self.l1, self.l2)
            if self.verbose:
                print lossf(self.w, X, y, self.l1, self.l2)
        return self
    
    
    def predict_proba(self, X):
        probs = 1 / (1 + np.exp(-sum((X*self.w).T)))
        return probs

    def predict(self, X):
        predicts = np.ones(len(self.predict_proba(X)))
        predicts[self.predict_proba(X) < 0.5] = -1 
        return predicts

def test_work():
    print ("Start test")
    X, y = make_classification(n_features=100, n_samples=1000)
    y = 2 * (y - 0.5)

    try:
        clf = LR(lr=1e-3, l1=1, l2=1e-4, num_iter=1000, verbose=0)
    except Exception:
        assert False, "Создание модели завершается с ошибкой"
        return

    try:
        clf = clf.fit(X, y)
    except Exception:
        assert False, "Обучение модели завершается с ошибкой"
        return

    assert isinstance(lossf(clf.w, X, y, 1e-3, 1e-3), float), "Функция потерь должна быть скалярной и иметь тип np.float"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).shape == (100,), "Размерность градиента должна совпадать с числом параметров"
    assert gradf(clf.w, X, y, 1e-3, 1e-3).dtype == np.float, "Вектор градиента, должен состоять из элементов типа np.float"
    assert clf.predict(X).shape == (1000,), "Размер вектора предсказаний, должен совпадать с количеством объектов"
    assert np.min(clf.predict_proba(X)) >= 0, "Вероятности должны быть не меньше, чем 0"
    assert np.max(clf.predict_proba(X)) <= 1, "Вероятности должны быть не больше, чем 1"
    assert len(set(clf.predict(X))) == 2, "Метод предсказывает больше чем 2 класса на двух классовой задаче"
    print "End tests"
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X, y)
    print lr.predict_proba(X).shape
    print sum(abs(lr.predict_proba(X)[:,1]-clf.predict_proba(X)))
    print clf.predict_proba(X)[1:50]
    print clf.predict(X)[1:50]

test_work()
