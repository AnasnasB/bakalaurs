import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def torso_f(signal):
    return 1
def ttl_bw(signal):
    return 2
def offst(signal):
    return 3
def noMD_bw(signal):
    return 4
def norm_std(signal):
    return 5
def limb_motion(signal):
    return 6

def extract_features(signal):
    torso_freq = torso_f(signal)
    total_bw = ttl_bw(signal)
    offset = offst(signal)
    bw_without_microdopplers = noMD_bw(signal)
    std_normalized = norm_std(signal)
    limb_motion_period = limb_motion(signal)
    return [torso_freq, total_bw, offset, bw_without_microdopplers, std_normalized, limb_motion_period]

mat_data = scipy.io.loadmat('./W/3m_0_file1.mat')
doppler_signal = np.transpose(mat_data['received_time_domain_signal'])

window_size = 256
step_size = int(window_size/2)
features = []
labels = ["walking"]
for i in range(0, len(doppler_signal) - window_size, step_size):
    window = doppler_signal[i:i + window_size]
    feature_vector = extract_features(window)
    features.append(feature_vector)
    labels.append(1)
print(features)