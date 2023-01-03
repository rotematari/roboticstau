# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import torch
import data_loader
import data_agmuntation
import paramaters
dirpath = paramaters.parameters.dirpath
items = paramaters.parameters.items

sample_rate = 10
x = data_loader.Data(train=True, dirpath=dirpath, items=items)



y1 = x.X[:, 0].numpy()
y2 = x.X[:, 1].numpy()
y3 = x.X[:, 2].numpy()
y4 = x.X[:, 3].numpy()
y5 = x.X[:, 4].numpy()
# y6 = x.X[:, 5].numpy()
lables = x.Y

# list(y)
plt.figure(1)
plt.plot(list(y1))
plt.plot(list(y2))
plt.plot(list(y3))
plt.plot(list(y4))
plt.plot(list(y5))
# plt.plot(list(y6))
plt.plot(list(lables))
plt.legend(items)


plt.figure(2)
z = data_loader.Data(train=False, dirpath=dirpath, items=items)



y1 = z.X[:, 0].numpy()
y2 = z.X[:, 1].numpy()
y3 = z.X[:, 2].numpy()
y4 = z.X[:, 3].numpy()
y5 = z.X[:, 4].numpy()
# y6 = x.X[:, 5].numpy()
lables = z.Y

# list(y)
plt.plot(list(y1))
plt.plot(list(y2))
plt.plot(list(y3))
plt.plot(list(y4))
plt.plot(list(y5))
# plt.plot(list(y6))
plt.plot(list(lables))
plt.legend(items)
plt.show()
