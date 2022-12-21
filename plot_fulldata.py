# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import emd
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
y6 = x.X[:, 5].numpy()
lables = x.Y

# list(y)
plt.figure(1)
plt.plot(list(y1))
plt.plot(list(y2))
plt.plot(list(y3))
plt.plot(list(y4))
plt.plot(list(y5))
plt.plot(list(y6))
plt.plot(list(lables))
plt.legend(items)
plt.show()
# print(y)
# imf = emd.sift.sift(y)
# IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
#
# # Define frequency range (low_freq, high_freq, nsteps, spacing)
# freq_range = (0.1, 10, 80, 'log')
# f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
#
# # print(imf)
# proto_imf = y.copy()
# # Compute upper and lower envelopes
# upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
# lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
# avg_env = (upper_env+lower_env) / 2
#
# # plt.figure()
# # plt.plot(y, 'k')
# # plt.plot(upper_env)
# # plt.plot(lower_env)
# # plt.plot(avg_env)
# # # plt.xlim(600, 1000)
# # plt.legend(['Signal', 'Upper Env', 'Lower Env', 'Avg Env'])
# # # emd.plotting.plot_imfs(imf)
# # plt.show()
#
# def emd_transform(feature_data, labele_data): # data from 1 class only
#     for i in len(feature_data[0]):
#         proto_imf = feature_data[:,i]
#
#         # Compute upper and lower envelopes
#         upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
#         lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
#         avg_env = (upper_env + lower_env) / 2
#
#     return
