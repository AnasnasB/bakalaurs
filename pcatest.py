import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
import os
import scipy.io
from scipy.fftpack import fftshift


def process_file_with_spectrogram_and_pca(file_path, fs, prf, SweepTime, fstart, fstop, NTS, Bw):
    # Load data from .mat file
    mat_data = scipy.io.loadmat(file_path)
    rawData = (np.transpose(mat_data['sx1'])).flatten()  # Adjust based on variable names in .mat file
    # Reshape data
    Colmn = len(rawData) // NTS
    rawData = rawData[:Colmn * NTS].reshape(NTS, Colmn)

    # FFT processing
    fftRawData = fftshift(np.fft.fft(rawData, axis=0), axes=0)
    rp = fftRawData[NTS // 2:, :]  # Range profile (only positive frequencies)

    # MTI filtering
    h = np.array([1, -2, 3, -2, 1])  # High-pass filter
    m, n = rp.shape
    rngpro = np.zeros((m, n + len(h) - 1))
    for k in range(m):
        rngpro[k, :] = np.convolve(h, np.abs(rp[k, :]), mode='full')

    # Extract a single row for the spectrogram (e.g., row 8)
    mdsignal = rngpro[8, :]

    # Compute the spectrogram
    f, t, Sxx = spectrogram(mdsignal, fs=prf, nperseg=256, noverlap=197, nfft=2**12)
    return t, f, 20* np.log10(np.abs(Sxx)/np.max(np.max(np.abs(Sxx))))


def main():
    # Radar and spectrogram parameters
    fstart = 77.1799e9  # Start Frequency (Hz)
    fstop = 77.9474e9  # Stop Frequency (Hz)
    fc = (fstart + fstop) / 2  # Center Frequency (Hz)
    c = 3e8  # Speed of light (m/s)
    lambda_c = c / fc  # Wavelength (m)
    SweepTime = 40e-3  # Sweep Time per frame (s)
    NTS = 256  # Number of time samples per sweep
    NPpF = 128  # Number of pulses per frame
    Bw = fstop - fstart  # Bandwidth (Hz)
    sampleRate = 10e6  # Sampling Rate (Hz)
    prf = 1 / (SweepTime / NPpF)  # Pulse repetition frequency

    path = './05_Walking_Towards_radar_2/04000000_1574696114_Raw_0_matrix.mat'
    t, f, Sxx = process_file_with_spectrogram_and_pca(
                path, sampleRate, prf, SweepTime, fstart, fstop, NTS, Bw
            )
    spectrogram_flattened = Sxx.T  # Transpose to make it [time x frequency]

    # Step 4: Apply PCA
    n_components = 5  # Number of principal components to retain
    pca = PCA(n_components=n_components)
    spectrogram_pca = pca.fit_transform(spectrogram_flattened)
    
    # Step 5: Visualize the results
    # Original spectrogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, f, Sxx)
    plt.title("Original Spectrogram")
    plt.ylabel('Frequency [Hz]')
    # plt.ylim(0, 200)
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]')
    
    # PCA-reduced spectrogram (first component as an example)
    plt.subplot(1, 2, 2)
    for i in range(n_components):
        plt.plot(t, spectrogram_pca[:, i], label=f'PC {i+1}')
    plt.title("PCA-Reduced Spectrogram")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Optional: Explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)


if __name__ == "__main__":
    main()