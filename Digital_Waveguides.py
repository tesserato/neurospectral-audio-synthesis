import numpy as np
import matplotlib.pyplot as plt
import os
from Hlib import save_wav
import time

path = 'Demo Digital Waveguide/'
n = 44100
fps = 44100
frequency = 440
amplitude = 20000
pluck_position = .5
pickup_position = .5
sustain = .99
smoothing = 3
plot = False

os.makedirs(path, exist_ok=True)
os.makedirs(path + 'plots', exist_ok=True)
L = int(round(fps / (2 * frequency)))
pickup = int(round(L * pickup_position))
asc = int(round(L * pluck_position))
dsc = L - asc
X_asc = np.linspace(0, 1, asc)
X_dsc = np.linspace(0, 1, dsc + 1)[::-1][1: ]
delay_r = np.hstack([X_asc,X_dsc])
delay_l = np.zeros(L)

initial_time = time.time()
w = np.zeros(n)
zeros = np.zeros(L)
for i in range(n):
  print('step ', i + 1,' of ', n)
  if plot and i <= 2000:
    fig = plt.figure(1)
    fig.set_size_inches(1000 / fig.dpi, 400 / fig.dpi)
    plt.subplot(311)
    plt.axis([0, L, -1.1, 1.1])
    plt.plot(delay_r,'.')
    plt.plot(zeros,'k')    
    plt.subplot(312)
    plt.axis([0, L, -1.1, 1.1])
    plt.plot(delay_l,'.')
    plt.plot(zeros,'k')
    plt.subplot(313)
    plt.axis([0, L, -1.1, 1.1])
    plt.plot(delay_r + delay_l,'k')
    plt.axvline(pickup)
    plt.savefig(path + 'plots' + '/' + str(i) + '.png', bbox_inches='tight')
    plt.close()
  w[i] = delay_r[pickup] + delay_l[pickup]
  to_l = -1 * np.average(delay_r[-smoothing:]) * sustain # to add in delay_l[L-1] AFTER rolling
  delay_r = np.roll(delay_r, 1)
  delay_r[0] = -1 * delay_l[0]
  delay_l = np.roll(delay_l, -1)
  delay_l[L-1] = to_l
print('time =', time.time() - initial_time)
save_wav(
  w * amplitude,
  path + str(frequency) +
  '_Pluck=' + str(pluck_position) +
  '_Pick=' + str(pickup_position) + '.wav'
  )

