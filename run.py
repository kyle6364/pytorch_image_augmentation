# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n = 200
p = 0.5
k = np.arange(0, 401)

binomial = stats.binom.pmf(k, n, p)

plt.plot(k, binomial, 'o-')
plt.title('binomial:n=%i,p=%.2f' % (n, p), fontsize=15)
plt.xlabel('number of success')
plt.ylabel('probability of success', fontsize=15)
plt.grid(True)
plt.show()
