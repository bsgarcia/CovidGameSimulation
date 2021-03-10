import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp


a = np.random.dirichlet([7, 1, 1], size=1)
import pdb; pdb.set_trace()
b = np.random.beta(1, 40, size=10**6)
label = range(1, 7)
for i in range(a.shape[1]):
    sns.distplot(a[:,i], label=label[i], hist_kws={'edgecolor': 'w'})

plt.legend()
plt.show()
