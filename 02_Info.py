from numpy.fft import rfft, irfft
import numpy as np
import matplotlib.pyplot as plt
from Hlib import read_wav, save_wav, create_signal
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
import os

def pad(X, n, value = 0):
  l = X.shape[0]
  if l >= n:
    padded = X[0:n]
  else:
    zeros = np.zeros(n - l) #.fill(np.nan)
    padded = np.hstack([X, zeros])
  # print('p:', padded.shape[0])
  return padded
##########################################
##########################################
################ Settings ################
piece_name = 'piano'
partials = 100
##########################################
##########################################
##########################################

filenames = os.listdir('00_samples/' + piece_name + '/')
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

save_path = '01_Info/' + piece_name + '/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + 'graphs/', exist_ok=True)

# extract frequencies, amplitudes and decays per partials
max_amps = []
amps_over_max_amp = []
decays = []
Mfreqs_over_Tfreqs = []
phases = []
Bs = []

Part = np.arange(1, partials + 1)
for name in names:
  print('\nsample:', name)
  s, fps = read_wav('00_samples/' + piece_name + '/' + name + '.wav')
  n = s.shape[0]
  F = rfft(s) * 2 / n
  P = np.abs(F)
  max_amp = np.max(P)                                              #<| <| <| <| <|
  max_amps.append(max_amp)
  k = int(name)
  f0 = np.power(2, (k - 49) / 12) * 440 * (1.17912446e-07 * k ** 3 -1.59501627e-05 * k ** 2 + 8.64681234e-04 * k + 9.86196582e-01)

  local_f0 = int(round(f0 / fps * n))
  idxs = argrelmax(P, order=local_f0 // 2)[0] #Local Frequencies
  idx0 = idxs[np.argmin(np.abs(idxs - local_f0))]
  idxs = idxs[np.where(idxs >= idx0)][:partials]
  fs_x_partials = idxs * fps / n                                   #<| <| <| <| <|
  as_x_partials = P[idxs]                                          #<| <| <| <| <|
  ps_x_partials = np.angle(F)                                      #<| <| <| <| <|
  X = pad(fs_x_partials, partials)
  Y = ((X / Part) ** 2 - f0 ** 2) / (f0**2 * Part ** 2)
  b, _, _, _ = np.linalg.lstsq((Part**2)[:,np.newaxis], Y)
  Bs.append(b)
  middle = int(round(n/2))
  x = np.arange(n)
  ds_x_partials = np.zeros(partials)
  for i, local_frequency in enumerate(idxs):
    y = np.exp(-2.0 * np.pi * local_frequency * x / n * 1j)
    Z = y * s
    a1 = np.abs(np.sum(Z[ : middle])) / middle
    a2 = np.abs(np.sum(Z[middle : ])) / middle
    ds_x_partials[i] = 2 * np.log(a1 / a2) / n                     #<| <| <| <| <|

  amps_over_max_amp.append(pad(as_x_partials, partials) / max_amp)
  decays.append(pad(ds_x_partials, partials))
  inharms = fs_x_partials / (f0 * np.arange(1, fs_x_partials.shape[0] + 1))
  Mfreqs_over_Tfreqs.append(pad(inharms, partials))
  phases.append(pad(ps_x_partials, partials))

  highest_local_freq = np.max(idxs)
  X = np.arange(highest_local_freq) * fps / n

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(X, P[ : highest_local_freq], '0.5')
  plt.plot(idxs * fps / n, P[idxs],'ro', label='Peaks')
  plt.legend()
  plt.savefig(save_path + 'graphs/01_peaks_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(amps_over_max_amp[-1], 'k.-')
  plt.savefig(save_path + 'graphs/02_amps_over_max_amp_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(decays[-1], 'k.-')
  plt.savefig(save_path + 'graphs/03_decays_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(Mfreqs_over_Tfreqs[-1], 'k.-')
  plt.savefig(save_path + 'graphs/04_Mfreqs_over_Tfreqs_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(Part**2, Y, 'k.-')
  plt.savefig(save_path + 'graphs/05_b_' + name + '.png')
  plt.close()

fig = plt.figure()
fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
plt.plot(max_amps, 'k.-')
plt.savefig(save_path + 'graphs/06_max_amps_' + name + '.png')
plt.close()

Mfreqs_over_Tfreqs = np.array(Mfreqs_over_Tfreqs)
amps_over_max_amp = np.array(amps_over_max_amp)
decays = np.array(decays)
phases = np.array(phases)
max_amps = np.array(max_amps)
Bs = np.array(Bs)

np.savetxt(save_path + 'Mfreqs_over_Tfreqs.csv', Mfreqs_over_Tfreqs, delimiter=',')
np.savetxt(save_path + 'amps_over_max_amp.csv', amps_over_max_amp, delimiter=',')
np.savetxt(save_path + 'decays.csv', decays, delimiter=',')
np.savetxt(save_path + 'phases.csv', phases, delimiter=',')
np.savetxt(save_path + 'max_amps.csv', max_amps, delimiter=',')
np.savetxt(save_path + 'Bs.csv', Bs, delimiter=',')

np.save(save_path + 'Mfreqs_over_Tfreqs.npy', Mfreqs_over_Tfreqs) 
np.save(save_path + 'amps_over_max_amp.npy', amps_over_max_amp)
np.save(save_path + 'decays.npy', decays)

