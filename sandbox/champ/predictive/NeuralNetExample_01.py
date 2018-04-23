
from sklearn.neural_network import MLPRegressor
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

clf = MLPRegressor(activation='relu',solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(50,), random_state=1,
                     early_stopping=False,momentum=0.9,
                     batch_size='auto')

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










