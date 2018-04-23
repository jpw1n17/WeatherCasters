
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# create sample data
# =============================================================================

n_samples, n_features = 100, 5
np.random.seed(0)
y = np.random.randn(n_samples)
x = np.random.randn(n_samples, n_features)

# =============================================================================
# create model
# =============================================================================

clf = Ridge(alpha=1.0)
clf.fit(x, y) 
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
clf.coef_

# =============================================================================
# train model
# =============================================================================

ytr = y[0:10]
xtr = x[0:10]

yts = y[10:100]
xts = x[10:100] 

clf.fit(xtr, ytr)                         
yhts = clf.predict(xts)

# =============================================================================
# Plot graph
# =============================================================================

plt.scatter(yts, yhts)
t = np.arange(-2, 2, 0.1)
plt.figure(1)
plt.plot(t, t)


# =============================================================================
# 
# =============================================================================






