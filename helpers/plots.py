import wave
import pylab
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from  .npdata import extract_frequencies
from  .files import get_genre, get_filename
from genre import Genre
from feature import SoundFeature 

HOP = 512
SAMP_RATE = 22050

def plot_wavesform(paths_to_tunes, begin=0, n_seconds=10):
    """ Plot the waveform of a few tunes """

    fig, axs = plt.subplots(len(paths_to_tunes), 1, figsize=(15,15))

    max_frame = int(begin * SAMP_RATE) + int(n_seconds * SAMP_RATE)
    time = np.arange(begin, begin + n_seconds, 1.0 / SAMP_RATE)

    for ax, tune in zip(axs.flat, paths_to_tunes):
        genre = get_genre(tune)
        
        freqs = extract_frequencies(tune, normalized=True)
        
        ax.plot(time, freqs[int(begin * SAMP_RATE):max_frame])
        ax.set_title(" Genre: " + genre, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Time (s) \n", fontsize=12)
        ax.set_ylabel("Amplitde (AU)", fontsize=12)
        ax.set_xlim([begin, begin + n_seconds])
        ax.set_ylim([-1, 1])
    
    fig.tight_layout()
    fig.savefig('waveforms.png')


# Superimpose feature on waveform
def plot_superimprose_wave(paths, feature, duration=None):
    fig, axs = plt.subplots(3, 2, figsize=(15, 8))

    for ax, tune in zip(axs.flat, paths):
        genre = get_genre(tune)

        freqs = extract_frequencies(tune, duration=duration, normalized=True)
        
        time = [x / float(SAMP_RATE) for x in range(len(freqs))]

        if feature == SoundFeature.ZCR:
            feat = librosa.feature.zero_crossing_rate(freqs)[0]
            times = [x * float(HOP) / SAMP_RATE for x in range(len(feat))]

        if feature == SoundFeature.RMS:
            feat = librosa.feature.rmse(freqs)[0]
            times = [x * float(HOP) / SAMP_RATE for x in range(len(feat))]

        if feature == SoundFeature.SC:
            feat = librosa.feature.spectral_centroid(freqs, sr=22050)[0]
            times = [x * float(HOP) / SAMP_RATE for x in range(len(feat))]

        max_feat = max(feat)
        feat_norm = [x / max_feat for x in feat]

        ax.plot(time, freqs, color='g', alpha=0.2)
        ax.set_title(feature.name + ": " + genre, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitde (AU)", fontsize=8)
        ax.set_xlim([0, time[-1]])
        ax.set_ylim([-1, 1])

        ax2 = ax.twinx()
        ax2.plot(times, feat_norm, color='r')
        ax2.tick_params(labelsize=8)
        ax2.set_ylabel(feature.name + ' (AU)', fontsize=8)
        ax2.set_xlim([0, time[-1]])
        ax2.set_ylim([-1, 1])

    fig.tight_layout()
    fig.savefig(feature.name + ' - superimprose - wave.png')

def plot_iqr(df, genres, feature, duration=20):
    df = df.sort_values(by=["genre"])

    feature_name = feature.name.lower()

    matrix_data = (df[feature_name].apply(pd.Series).as_matrix())

    x2=np.percentile(matrix_data,75,axis=1)-np.percentile(matrix_data,25,axis=1)

    a=[]
    for i in range(5):    
        a.append(x2[i*100:(i+1)*100])

    fig,ax=plt.subplots(figsize=(14,6))
    ax.boxplot(a,labels=genres, sym='r+')
    ax.set_title(f'inter-quartile range of {feature_name}(s) of audio signals in five genres')
    fig.savefig(f'inter-quartile-{feature_name}.png', bbox_inches='tight')

def plot_spectrograms(paths, duration=-1):
    fig, axs = plt.subplots(len(paths), 1, figsize=(15,15))

    for ax, tune in zip(axs.flat, paths):

        song_genre = get_genre(tune)
        song_name = get_filename(tune)

        freqs = extract_frequencies(tune, duration=duration)

        seconds_lmit = len(freqs) / SAMP_RATE

        ax.specgram(freqs, Fs=SAMP_RATE, cmap='gist_ncar')
        ax.set_title(f"{song_genre}:\n {song_name}", fontsize=12)

        ax.tick_params(labelsize=10)
        ax.set_xlim([0, seconds_lmit])
        ax.set_ylim([0, SAMP_RATE / 2])
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)

    fig.tight_layout()
    fig.savefig('spectrograms.png')

