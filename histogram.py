# %%

import matplotlib.pyplot as plt
import numpy as np
from  numpy.random import normal

np.random.seed(12345)

cat1 = normal(0, 1, 50)
cat2 = normal(0, 1, 20)
cat3 = normal(0, 1, 60)


add1 = normal(0, 1, 50)
add2 = normal(0, 1, 20)
add3 = normal(0, 1, 60)

bns = np.linspace(-4,4,10)

fig, axs = plt.subplots(3, figsize = (10,10))
axs[0].hist([cat1, cat2, cat3], bins = bns,   alpha = 0.6)
axs[1].hist([cat1+add1, cat2+add2, cat3+add3], bins = bns, alpha = 0.6)
axs[2].hist([cat1+0.5, cat2, cat3-0.5], bins = bns, alpha = 0.6)
axs[0].vlines((np.mean(cat1), np.mean(cat2), np.mean(cat3)), 0,45, color= ["blue", "orange", "green"])
axs[1].vlines((np.mean(cat1+add1), np.mean(cat2+ add2), np.mean(cat3 + add3)), 0,45, color= ["blue", "orange", "green"])
axs[2].vlines((np.mean(cat1)+0.5, np.mean(cat2), np.mean(cat3)-0.5),  0,45, color= ["blue", "orange", "green"])
plt.setp(axs, xlim = (-4,4), ylim = (0,25))
plt.show()

# %%
