import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

a = 10
b = 10
x = np.linspace(0, 1, 100)
opponent_disclosed = True
learning_rate = 20
plt.plot(x, stats.beta.pdf(x, a=a, b=b+opponent_disclosed*learning_rate))
plt.ylim([-.08, 5])
plt.title('Contribution PDF')
plt.show()