import keras as K
import numpy as np
from Hlib import save_wav, create_signal
import time

def scaled_tanh(x):
  return K.activations.tanh(6 * x - 3) / 2 + 1 / 2


piece_name = "piano"
key_name = "49"
partials = 100
n = 44100 * 10
fps = 44100

P = np.arange(1, partials + 1, 1)

[[mf, ma, md], [Mf, Ma, Md]] = np.genfromtxt('03_train/' + piece_name + '/m_Ms.csv', delimiter=',')

net = K.models.load_model('03_train/' + piece_name + '/#model.h5', {'scaled_tanh': scaled_tanh})

K = np.genfromtxt("01_tuning_stretch/" + piece_name + "/coefs.csv")


st = time.time() # <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> #

key = int(key_name)
f0 = np.power(2, (key - 49) / 12) * 440 * (K[0] * key ** K[1] * key ** 2 + K[2] * key + K[3])
key = (key - 1) / 87
ipt = np.array([[key, p] for p in np.linspace(0, 1, partials)])


[F,A,D] = net.predict(ipt)

F = (np.ndarray.flatten(F) * (Mf - mf) + mf) * P * f0
A = np.ndarray.flatten(A) * ((Ma - ma) + ma) * 5000
D = np.ndarray.flatten(D) * ((Md - md) + md)


# print(A.shape,D.shape,F.shape, P.shape)
w = np.zeros(n)
for i in range(partials):
  # print('partial:', i)
  ph = np.random.uniform(0, 2 * np.pi)
  w += create_signal(n, fps, F[i], ph, A[i], D[i])


print(time.time() - st) # <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> #

save_wav(w, key_name + ".wav")