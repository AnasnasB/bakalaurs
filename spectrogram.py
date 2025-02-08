import scipy.io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft, windows
from scipy.io import loadmat

file_path = "./05_Walking_Towards_radar_2/04000000_1574696114_Raw_0_matrix.mat"
mat_data = scipy.io.loadmat(file_path)
print(mat_data.keys())

# mdsignal = (np.transpose(mat_data['sx1'])).flatten()
mdsignal = (np.transpose(mat_data['sx1'])).flatten()
fs = 10**6 
nperseg = 256 
noverlap = 197
#20*log10(abs(flipud(sx1)/max(max(abs(sx1)))))
plt.figure(figsize=(10, 6))

frequencies, times, Sxx = spectrogram(mdsignal, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)

plt.specgram(20* np.log10(np.abs(mdsignal)/np.max(np.max(np.abs(mdsignal)))), NFFT=nperseg, Fs=fs, noverlap=noverlap, scale_by_freq=True,  sides='twosided', vmin=-75, vmax=0)
plt.title("Micro-Doppler Spectrogram")
plt.ylabel("Frequency (Hz)")
# plt.ylim(-10000, +10000)
plt.xlabel("Time (s)")
plt.colorbar(label="Power (dB)")
plt.show()

