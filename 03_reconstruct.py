import numpy as np
import matplotlib.pyplot as plt
from Hlib import read_wav, save_wav, create_signal
import os


piece_name = 'piano'
fps = 44100
n = fps * 10
save_path = '03_waves/' + piece_name + '/'
os.makedirs(save_path, exist_ok=True)


partials = 100
# filenames = os.listdir('00_samples/' + piece_name + '/')
# names = []
# for filename in filenames:
#   names.append(filename.replace('.wav', ''))
# names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

names = []
for i in range(1, 88 + 1):
  names.append(str(i))



A = np.genfromtxt('02_predictions/' + piece_name + '/amps_over_max_amp.csv', delimiter=',')
D = np.genfromtxt('02_predictions/' + piece_name + '/decays.csv', delimiter=',')
I = np.genfromtxt('02_predictions/' + piece_name + '/Mfreqs_over_Tfreqs.csv', delimiter=',')

for i, name in enumerate(names):
  k = int(name)
  print(name)
  w = np.zeros(n)
  f0 = np.power(2, (k - 49) / 12) * 440 * (1.17912446e-07 * k ** 3 -1.59501627e-05 * k ** 2 + 8.64681234e-04 * k + 9.86196582e-01)
  for j in range(partials):
    p = np.random.uniform(0, 2 * np.pi)
    f = f0 * (j+1) * I[i, j]
    d = D[i, j]
    a = A[i, j] * 5000
    w += create_signal(n, fps, f, p, a, d)

  save_wav(w, save_path + name + '.wav')
