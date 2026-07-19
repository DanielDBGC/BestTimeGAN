import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.utils.config import WANTED_CHANNELS

def plot_psd_comparison(h5_path, fs=250.0):
    """
    Loads EEG data and plots the average PSD for each stimulus frequency.
    """
    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        X = f['trials'][:]
        y = f['labels'][:]
    
    # Map integer labels back to stimulus frequencies
    # Assuming default: 4, 8, 12, ..., 60 Hz
    stim_freqs = list(range(4, 61, 4))
    
    unique_labels = np.unique(y)
    
    plt.figure(figsize=(15, 10))
    
    # We will average the PSD across all trials and all channels for each class
    for label in unique_labels:
        # Get all trials for this class
        X_class = X[y == label]
        
        # Calculate PSD for each trial and channel
        # X_class shape: (n_trials, n_samples, n_channels)
        n_trials, n_samples, n_channels = X_class.shape
        
        # We will calculate the PSD across the sample dimension (axis=1)
        freqs, psds = welch(X_class, fs=fs, nperseg=512, axis=1)
        
        # Average PSD across trials and channels
        mean_psd = np.mean(psds, axis=(0, 2))
        
        stim_freq = stim_freqs[label]
        plt.plot(freqs, mean_psd, label=f'{stim_freq} Hz', linewidth=1.5)
        
        # Optional: Add a vertical dashed line at the stimulus frequency to see if it matches the peak
        plt.axvline(x=stim_freq, color='gray', linestyle='--', alpha=0.3)

    plt.title('Power Spectral Density (PSD) Comparison by Stimulus Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.xlim(2, 65)  # Limit x-axis to the range of interest (stim frequencies are 4-60)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_file = 'psd_comparison.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_file}!")
    plt.show()

if __name__ == "__main__":
    # You can change this to point to a different dataset if you want
    data_path = 'data/processed/eeg_test_8.h5'
    
    # Note: Using fs=250.0 Hz by default. Update if your EDF files have a different sampling rate.
    plot_psd_comparison(data_path, fs=1000.0)
