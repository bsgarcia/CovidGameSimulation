import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as sp
import arviz as az
# import pymc3 as pm


alphas = np.ones(10)
alphas[2] = 2
alphas[8] = 5
alphas[7] = 10
alphas[9] = 30
a = np.random.dirichlet(alphas, size=99999)

dd = {}
label = range(1, 11)
for i in label:
    dd[f'C={i}'] = a[:, i - 1]

az.plot_forest(dd, kind='ridgeplot', ridgeplot_overlap=7, ridgeplot_alpha=.7,
               figsize=(10, 5), ridgeplot_truncate=False, combined=True, colors='cycle')
plt.xlim([0,1])

plt.show()
