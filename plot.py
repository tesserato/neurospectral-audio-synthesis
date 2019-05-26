import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 30})



piece_name = 'piano'
path = '02_predictions/' + piece_name + '/'
decays = np.genfromtxt('01_info/' + piece_name + '/fractions_of_max_decays.csv', delimiter=',')
partials = decays.shape[1]

frequencies = np.genfromtxt('01_info/' + piece_name + '/Mfreqs_over_Tfreqs.csv', delimiter=',')
partials = frequencies.shape[1]
amplitudes = np.genfromtxt('01_info/' + piece_name + '/fractions_of_max_amps.csv', delimiter=',')
decays = np.genfromtxt('01_info/' + piece_name + '/fractions_of_max_decays.csv', delimiter=',')


pred_frequencies = np.genfromtxt('02_predictions/' + piece_name + '/Mfreqs_over_Tfreqs.csv', delimiter=',')
pred_amps = np.genfromtxt('02_predictions/' + piece_name + '/fractions_of_max_amps.csv', delimiter=',')
pred_decays = np.genfromtxt('02_predictions/' + piece_name + '/fractions_of_max_decays.csv', delimiter=',')


X = []
for name in range(1, 88 + 1, 1):
  for p in np.arange(0, partials):
    X.append([p])
X = np.array(X)

X = X.reshape(-1, partials)

fig = plt.figure()
fig.set_size_inches(1500 / fig.dpi, 1200 / fig.dpi)
plt.tight_layout()
nonzero_idxs = np.where(frequencies > 0)
plt.plot(X[nonzero_idxs], frequencies[nonzero_idxs],'kx')
plt.plot(X, pred_frequencies,'r.')
plt.savefig(path + '_Mfreqs_over_Tfreqs.png')
plt.close()

fig = plt.figure()
fig.set_size_inches(1500 / fig.dpi, 1200 / fig.dpi)
plt.tight_layout()
plt.plot(X, amplitudes ,'kx')
plt.plot(X, pred_amps ,'r.')
plt.savefig(path + '_fractions_of_max_amps.png')
plt.close()

fig = plt.figure()
fig.set_size_inches(1500 / fig.dpi, 1200 / fig.dpi)
plt.tight_layout()
max_decays = np.genfromtxt('01_Info\piano\max_decays.csv', delimiter=',')
max_decays = np.tile(max_decays, partials)
max_decays = np.reshape(max_decays, (-1, partials))
print(X.shape, max_decays.shape)

nonzero_idxs = np.where(decays > 0)
plt.plot(X[nonzero_idxs], decays[nonzero_idxs] * max_decays[nonzero_idxs],'kx')
plt.plot(X, pred_decays * max_decays,'r.')
plt.savefig(path + '_fractions_of_max_decays.png')
plt.close()