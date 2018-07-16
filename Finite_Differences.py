import numpy as np
import matplotlib.pyplot as plt
import os
from Hlib import save_wav
import time

path = 'Demo Finite Difference/'
amplitude = 0.005 # meters
pluck_position = .1 # fraction of L
pickup_position = .1 # fraction of L
fps = 44100 # samples / second
frequency = 440 #hz
duration = 1 # seconds
L = 0.6 # meters
sustain = .9998
plot = True

os.makedirs(path, exist_ok=True)
os.makedirs(path + 'plots', exist_ok=True)
N = int(duration * fps) # number of time points
dt = 1 / fps # spacing beetwen discrete time points
M = int(fps / (2 * frequency)) # number of position points
dx = L / M # spacing beetwen discrete position points
c = frequency * 2 * L # meters / second
C = c * dt / dx # Courant number
x = np.arange(0, M + 1, 1) * dx
t = np.arange(0, N + 1, 1) * dt
pickup = int(round(M * pickup_position))
asc = int(round(M * pluck_position))
dsc = M + 2 - asc
X_asc = np.linspace(0, 1, asc)
X_dsc = np.linspace(0, 1, dsc)[::-1][1: ]
y = np.zeros(M + 1)
y_1 = np.hstack([X_asc,X_dsc]) * amplitude * np.random.normal(1, .01, M + 1) # initial displacement shape
y_2 = np.zeros(M + 1)

print(C, M, N)
initial_time = time.time()
ctr=0
w = []
for i in range(1, M):
  y[i] = y_1[i] + 0.5 * C**2 *(y_1[i+1] - 2*y_1[i] + y_1[i-1])
y[0] = 0
y[M] = 0
w.append(y[pickup])
y_2[:] = y_1
y_1[:] = y
if plot:
  fig = plt.figure(1)
  fig.set_size_inches(1000 / fig.dpi, 400 / fig.dpi)
  plt.plot(x, y, 'k')
  plt.axis([0, L, -1.1 * amplitude, 1.1 * amplitude])
  plt.axvline(L * pickup_position)
  plt.savefig(path + 'plots/' + str(ctr) + '.png', bbox_inches='tight')
  plt.close()
ctr += 1
for j in range(1, N):
  print('step ', ctr,' of ', N)
  for i in range(1, M):
    y[i] = 2 * y_1[i] - y_2[i] + C**2 * (y_1[i+1] - 2*y_1[i] + y_1[i-1])
  y[0] = y[0] * 0.5
  y[M] = y[M] * 0.5
  y = y * sustain
  w.append(y[pickup])
  y_2[:] = y_1 
  y_1[:] = y
  if plot and j <= 2000:
    fig = plt.figure(1)
    fig.set_size_inches(1000 / fig.dpi, 400 / fig.dpi)
    loop = 0
    plt.plot(x, y, 'k')
    plt.axis([0, L, -1.1 * amplitude, 1.1 * amplitude])
    plt.axvline(L * pickup_position)
    plt.savefig(path + 'plots/' + str(ctr) + '.png', bbox_inches='tight')
    plt.close()
  ctr += 1
w = np.array(w) * 4000000
print('time =', time.time() - initial_time)
save_wav(
  w,
  path + str(frequency) + 
  '_Pluck=' + str(pluck_position) + 
  '_Pick=' + str(pickup_position) + '.wav'
  )