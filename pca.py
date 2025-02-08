import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
import os
import scipy.io

fs = 1000
path = './05_Walking_Towards_radar_2'
signals=[]
for file in os.listdir(f"{path}"):
    if file.endswith('.mat'):
        mat_data = scipy.io.loadmat(f"{path}/{file}")
        signal=np.transpose(mat_data['sx1']).flatten()
 
        # Step 2: Compute the spectrogram
        fstart = 77.1799e7  # Start Frequency (Hz)
        fstop = 77.9474e7  # Stop Frequency (Hz)
        fc = (fstart + fstop) / 2  # Center Frequency (Hz)
        c = 3e8  # Speed of light (m/s)
        lambda_c = c / fc  # Wavelength (m)
        SweepTime = 40e-3  # Sweep Time per frame (s)
        NTS = 256  # Number of time samples per sweep
        NPpF = 128  # Number of pulses per frame
        Bw = fstop - fstart  # Bandwidth (Hz)
        sampleRate = 10e6  # Sampling Rate (Hz)
        prf = 1 / (SweepTime / NPpF)  # Pulse repetition frequency

        frequencies, times, Sxx = spectrogram(signal, sampleRate, return_onesided=False)
        # Step 3: Flatten the spectrogram for PCA
        spectrogram_flattened = Sxx.T  # Transpose to make it [time x frequency]
        # Step 4: Apply PCA
        n_components = 4  # Number of principal components to retain
        pca = PCA(n_components=n_components)
        spectrogram_pca = pca.fit_transform(spectrogram_flattened)
        plt.figure(figsize=(12, 5))
        for i in range(n_components):
            plt.plot(times, spectrogram_pca[:, i], label=f'PC {i+1}')
        plt.title("PCA-Reduced Spectrogram")
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(0.05,0.14)
        plt.tight_layout()
        plt.figtext(0.5, 0.01, f"Explained Variance Ratio: {pca.explained_variance_ratio_}", ha='center', va='center', fontsize=10)
        plt.savefig(f'./pictures_pca/{file}_pca_spectrogram.png')
        plt.clf()

        explained_variance_file = f'./pca_data/{file}_explained_variance.npy'
        np.save(explained_variance_file, pca.explained_variance_ratio_)
        pca_file = f'./pca_data/{file}_pca_components.npy'
        np.save(pca_file, spectrogram_pca)
        time_file = f'./pca_data/{file}_pca_time.npy'
        np.save(time_file, times)
 