
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


mat_data = scipy.io.loadmat('./W/3m_0_file1.mat')
signal = ((mat_data['received_time_domain_signal'])).flatten()

print(mat_data.keys())
fs = 10**3 
nperseg = 128 
noverlap = int(nperseg/2)

#normalizācija
frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)



# frequencies, times, Sxx = spectrogram(signal, fs=fs, scaling="spectrum")
# print(frequencies.min(), frequencies.max())
plt.figure(figsize=(10, 6))
plt.specgram(signal, NFFT=nperseg, Fs=fs, noverlap=noverlap, scale_by_freq=True,  sides='twosided')
plt.title("Micro-Doppler Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label="Power (dB)")
# plt.pcolormesh(times, frequencies, Sxx, shading='gouraud',vmax=0.3e-6)
# plt.title("Micro-Doppler Spectrogram")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Time (s)")
# plt.colorbar(label="Power")
plt.xlim(0, 100)
# plt.ylim(-fs/2, fs/2) 
plt.show()