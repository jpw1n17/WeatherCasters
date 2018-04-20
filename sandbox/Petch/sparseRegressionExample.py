# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:59:44 2018

@author: JaZz-
"""

from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y) 
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
clf.coef_


x2 = np.random.randn(n_samples, n_features)
y2 = clf.predict(X)

plt.scatter(y, y2)

t = np.arange(0, 5, 0.1)

plt.figure(1)

plt.plot(t, t)




n_samples, n_features = 100, 5
np.random.seed(0)
