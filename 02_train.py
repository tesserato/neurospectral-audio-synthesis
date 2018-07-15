import numpy as np
import keras as K
import matplotlib.pyplot as plt
# from Hlib import save_wav
from Hlib import save_model
from Hlib import normalize_cols, denormalize_cols
from Hlib import create_signal
import os

def scaled_tanh(x):
  return K.activations.tanh(6 * x - 3) / 2 + 1 / 2

#loss: 0.0169 - frequency_3_loss: 0.0016 - amplitude_3_loss: 0.0120 - decay_3_loss: 0.0033
piece_name = 'piano'
path = '02_predictions/' + piece_name + '/'
epcs = 5000
os.makedirs(path, exist_ok=True)

# reading and sorting filenames in given folder
filenames = os.listdir('00_samples/' + piece_name + '/')
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

# reading targets
orig_frequencies = np.load('01_info/' + piece_name + '/Mfreqs_over_Tfreqs.npy')
orig_amplitudes = np.load('01_info/' + piece_name + '/amps_over_max_amp.npy')
orig_decays = np.load('01_info/' + piece_name + '/decays.npy')

orig_decays = np.where(orig_decays >= 0, orig_decays, np.average(orig_decays))


# infer number of partials; check if it's the same for all quantities
partials = 0
if orig_frequencies.shape[1] == orig_amplitudes.shape[1] and orig_frequencies.shape[1] == orig_decays.shape[1]:
  partials = orig_frequencies.shape[1]
else:
  print("!")

frequencies = np.reshape(orig_frequencies, (-1, 1))
amplitudes = np.reshape(orig_amplitudes, (-1, 1))
decays = np.reshape(orig_decays, (-1, 1))

# preparing inputs
ipt = []
for name in names:
  key = (float(name) - 1) / 87
  for p in np.linspace(0, 1, partials):
    ipt.append([key, p])
ipt = np.array(ipt)


nonzero_idxs = np.where(frequencies > 0)

# removing zeros from tgt, and the respective inputs in ipt
frequencies = frequencies[nonzero_idxs[0]]
amplitudes = amplitudes[nonzero_idxs[0]]
decays = decays[nonzero_idxs[0]]
ipt = ipt[nonzero_idxs[0], :]


# orig_frequencies = np.reshape(frequencies, (-1, partials))
# orig_amplitudes = np.reshape(amplitudes, (-1, partials))
# orig_decays = np.reshape(decays, (-1, partials))

Mf = np.max(frequencies)
mf = np.min(frequencies)
frequencies = (frequencies - mf) / (Mf - mf)

Ma = np.max(amplitudes)
ma = np.min(amplitudes)
amplitudes = (amplitudes - ma) / (Ma - ma)

Md = np.max(decays)
md = np.min(decays)
decays = (decays - md) / (Md - md)

np.savetxt(path + 'm_Ms.csv', np.array([[mf,ma,md],[Mf,Ma,Md]]), delimiter=',')

print(partials, ipt.shape, nonzero_idxs[0].shape)

# preparing complete inputs
complete_ipt = []
for name in range(1, 88 + 1, 1):
  key = (float(name) - 1) / 87
  for p in np.linspace(0, 1, partials):
    complete_ipt.append([key, p])
complete_ipt = np.array(complete_ipt)

print(ipt.shape, frequencies.shape, amplitudes.shape, decays.shape)


#
# X = np.reshape(complete_ipt[:, 1], (-1, partials))
# plt.plot(orig_amplitudes.T,'k.')
# plt.show()
# exit()


act = scaled_tanh
init = 'glorot_normal'   #BESTFIT: 0.0198
# |> shared <| #
input = K.layers.Input(batch_shape=(None, ipt.shape[1]))

shared_layer = K.layers.Dense(
  units = 10,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='shared_layer_1'
)(input)

# shared_layer = K.layers.Dropout(0.1)(shared_layer)

shared_layer = K.layers.Dense(
  units = 10,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='shared_layer_2'
)(shared_layer)

# |> frequency <| #
f_layer = K.layers.Dense(
  units = 10,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='frequency_1'
)(shared_layer)

f_layer = K.layers.Dense(
  units = 10,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='frequency_2'
)(f_layer)

f_output = K.layers.Dense(
  units = frequencies.shape[1],
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='frequency_3'
)(f_layer)

# |> amplitude <| #
a_layer = K.layers.Dense(
  units = 70,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='amplitude_1'
)(shared_layer)

a_layer = K.layers.Dense(
  units = 70,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='amplitude_2'
)(a_layer)

a_output = K.layers.Dense(
  units = amplitudes.shape[1],
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='amplitude_3'
)(a_layer)

# |> decay <| #
d_layer = K.layers.Dense(
  units = 60,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='decay_1'
)(shared_layer)

d_layer = K.layers.Dense(
  units = 60,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='decay_2'
)(d_layer)

d_output = K.layers.Dense(
  units = decays.shape[1],
  activation=act,
  kernel_initializer=init,
  bias_initializer=init, 
  name='decay_3'
)(d_layer)

# output = K.layers.Add()([f_output, a_output, d_output])

model = K.models.Model(input, [f_output, a_output, d_output])
K.utils.print_summary(model)
model.compile(loss='mean_squared_error', optimizer=K.optimizers.nadam())
tb = K.callbacks.TensorBoard(path)
history = model.fit(ipt, [frequencies, amplitudes, decays], batch_size=ipt.shape[0], epochs=epcs, verbose=1, callbacks=[tb], shuffle=True)

save_model(model, path, 'model')


[pred_frequencies, pred_amplitudes, pred_decays] = model.predict(complete_ipt)

pred_frequencies = np.reshape(pred_frequencies,(-1, partials)) * (Mf - mf) + mf
pred_amplitudes = np.reshape(pred_amplitudes,(-1, partials)) * (Ma - ma) + ma
pred_decays = np.reshape(pred_decays,(-1, partials)) * (Md - md) + md

np.savetxt(path + 'Mfreqs_over_Tfreqs.csv', pred_frequencies, delimiter=',')
np.savetxt(path + 'amps_over_max_amp.csv', pred_amplitudes, delimiter=',')
np.savetxt(path + 'decays.csv', pred_decays, delimiter=',')

plt.plot(orig_frequencies.T ,'ko')
plt.plot(pred_frequencies.T ,'r.')
plt.savefig(path + '01_Mfreqs_over_Tfreqs.png')
plt.close()

plt.plot(orig_amplitudes.T ,'ko')
plt.plot(pred_amplitudes.T ,'r.')
plt.savefig(path + '02_amps_over_max_amp.png')
plt.close()

plt.plot(orig_decays.T ,'ko')
plt.plot(pred_decays.T ,'r.')
plt.savefig(path + '03_decays.png')
plt.close()


# X = np.reshape(complete_ipt[:, 1], (-1, partials))
# plt.plot(X, decays ,'ko')
# plt.plot(X, pred ,'r.')
# plt.savefig(path + 'fractions_of_max_decays.png')
# plt.close()


