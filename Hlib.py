import numpy as np
import matplotlib.pyplot as plt
import wave
import os
from scipy.fftpack import fft
from scipy.optimize import curve_fit


# class sample: # reads a wav sample to a sample object
#     def __init__ (self , path):
#         self.sound = wave.open(path , 'r')
#         self.signal = np.fromstring(self.sound.readframes(-1) , 'Int16')
#         self.fps = self.sound.getframerate()
#         self.T = len(self.signal) / self.fps
#         return

#     def get_signal(self):
#         T = np.linspace(0 , len(self.signal) / self.fps , len(self.signal) , endpoint=True)
#         Y = self.signal
#         return T, Y


def read_wav(path): # returns signal & fps
  wav = wave.open(path , 'r')
  signal = np.fromstring(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps

def save_wav(signal, name = 'test.wav', fps = 44100): #save .wav file to program folder
    o = wave.open(name, 'wb')
    o.setframerate(fps)
    o.setnchannels(1)
    o.setsampwidth(2)
    o.writeframes(np.int16(signal)) # Int16
    o.close()

def normalize_signal(signal, amin = 0, amax = 1):# returns amplitude also
    max_of_signal = max(np.abs(signal))
    return (signal / max_of_signal).tolist(), max_of_signal

def create_signal(N, fps, real_frequency, phase = 0, amplitude = 1, decay = 0):
  frequency = real_frequency * N / fps
  f = lambda x: amplitude * np.exp(-decay * x) * np.cos(phase + 2 * np.pi * frequency * x / N)
  X = np.arange(0,N,1)
  return f(X)

def save_model(model, path, name): # saves a Keras model
  import keras as K
  if not os.path.exists(path):
    os.makedirs(path)
  K.utils.plot_model(model, path + '#' + name + '.png', show_shapes=True)
  model_json = model.to_json()
  with open(path + '#' + name + '.json', 'w') as json_file:
    json_file.write(model_json)
  with open(path + '#' + name + '.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
  model.save(path + '#' + name + '.h5')

def decompose_DFT(f, terms, plot_Transform = False, verbose = False):
  f = np.array(f, dtype='float64')
  N = f.shape[0]
  # print('N =', N)
  F = fft(f) * 2 / N
  # fr = ifft(F)
  # print(np.allclose(f,fr))
  F = F[0 : N // 2 + 1]
  # print('F.shape =', F.shape)
  if plot_Transform:
    plt.plot(F.real,'o')
    plt.plot(F.imag,'o')
    plt.show()
  freqs = np.argsort(np.absolute(F))[::-1] # [::-1] reverts the array
  recomposed_signal = np.zeros(N, dtype='float64')
  errors = []
  for freq in freqs[:terms]:
    z = F[freq]
    amplitude = np.absolute(z)
    phase = np.angle(F[freq], False)
    recomposed_signal += create_signal(N, freq, phase, amplitude)
    loss = np.average((f - recomposed_signal) ** 2)
    errors.append(loss)
    if verbose:
      print('Freq.:', freq , 'Amp.:', round(amplitude, 2), 'Phase:', round(phase, 2), 'Error:', round(loss, 2), 'z:', round(z,2))
      print('---')
  return recomposed_signal

# def get_sinusoid(f, DECAY = True):
#   abs_f = np.abs(f)
#   sorted_f = -np.sort(-np.abs(f))
#   n = f.shape[0]
#   F = fft(f)
#   F = F[0 : n // 2 + 1]
#   freq = np.argmax(np.absolute(F))
#   amp = sorted_f[0]
#   summ = np.sum(abs_f)
#   dec = amp / summ
#   Y = lambda x: amp * np.exp(- dec * x)
#   X = np.arange(0, n, 1)
#   R = Y(X).astype('int')
#   end = int(np.argwhere(R == 0)[0])
#   phase = np.angle(F[freq], False)
#   if not DECAY:
#     dec = 0
#   recomposed_signal = create_signal(n, freq, phase, amp, dec)
#   return recomposed_signal, end, freq, phase, amp, dec

# def fit_function(x, frequency, phase, amplitude, decay):
#   N = x.shape[0]
#   return amplitude * np.exp(-decay * x) * np.cos(phase + 2 * np.pi * frequency * x / N)

def __fit_function(x, frequency, phase, amplitude, decay):  
  frequency = np.dtype('float64').type(frequency / 100)
  phase = np.dtype('float64').type(phase)
  amplitude = np.dtype('float64').type(amplitude / 1000)
  decay = np.dtype('float64').type(decay / 0.0001)
  N = x.shape[0]
  return amplitude * 1000 * np.exp(decay * 0.0001 * (-x)) * np.cos(phase + 2 * np.pi * 100 * frequency * x / N)

def get_sinusoid(sig, compute_decay = True):
  n = sig.shape[0]
  F = fft(sig) * 2 / n
  F = F[0 : n // 2 + 1]
  f = np.argmax(np.absolute(F))
  p = np.angle(F[f], False)
  a = np.absolute(F[f])
  s = np.sum(sig ** 2)
  d = (a**2 + a**2 * np.cos(2*p)) / (12 * s)
  print('PRE:', f, p, a, d)
  end = None
  X = np.arange(0, n, 1)
  bounds = ([-np.inf,-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf])
  popt, _ = curve_fit(__fit_function, X, sig, [f, p, a, d],method='trf', bounds=bounds)
  [f, p, a, d] = popt
  
  if not compute_decay:
    d = 0
  recomposed_sig = create_signal(n, f, p, a, d)
  loss = np.average((sig - recomposed_sig) ** 2)
  print('POS:',f,p,a,d, loss)
  return recomposed_sig, f, p, a, d

# def normalize_cols(matrix, m = -0.99, M = 0.99):
#   nmatrix = []
#   maxs = []
#   mins = []
#   for col in matrix.T:
#     ncol = []
#     cM = max(col)
#     maxs.append(cM)
#     cm = min(col)
#     mins.append(cm)
#     for i in col:
#       ncol.append(m + (M - m) * ((i - cm) / (cM - cm)))
#     nmatrix.append(ncol)
#   return np.array(nmatrix).T, np.array(maxs), np.array(mins)

def normalize_cols(matrix, m=0.0, M=1.0):
    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)
    rng = maxs - mins
    scaled_matrix = np.where(rng != 0, M - (((M - m) * (maxs - matrix)) / rng), 0)
    return scaled_matrix, maxs, mins

def denormalize_cols(scaled_matrix,mins, maxs, m=0.0, M=1.0):
  rng = maxs - mins
  matrix = maxs - (M - scaled_matrix) * rng / (M - m)
  return matrix

# def standardize_cols(matrix):
#   avgs = np.average(matrix, axis=0)
#   stds = np.std(matrix, axis=0)
#   standardized_matrix = (matrix - avgs) / stds

# def denormalize_cols(matrix, maxs, mins, m = -0.99, M = 0.99):
#   nmatrix = []
#   for idx, col in enumerate(matrix.T):
#     ncol = []
#     cM = maxs[idx]
#     cm = mins[idx]
#     for i in col:
#       ncol.append(((cM * m) - (cm * M) + (cm * i) - (cM * i))/(m-M))
#     nmatrix.append(ncol)
#   return np.array(nmatrix).T

def faf(signal, frequency): #Fourier Arbitrary Frequency
  n = signal.shape[0]
  x = np.arange(0, n, 1, dtype=np.int)
  y = np.exp(-2 * np.pi * x * frequency / n * 1j)
  z = np.sum(np.multiply(signal, y))
  phase = np.angle(z, False)
  return phase



def split_signal(s, steps = 10, name = 'signal'):
  print('Splitting:')
  errors = []
  parameters = []
  x = np.zeros(s.shape[0])
  r = s
  step = 0
  while step < steps:
    p, freq, phase, amp, dec = get_sinusoid(r)
    x += p
    r = s - x
    if dec >= 0:
      step += 1
      loss = np.average((s - x) ** 2)
      parameters.append(freq)
      parameters.append(phase)
      parameters.append( amp)
      parameters.append( dec)
      errors.append([loss])
  return x, r, parameters, errors