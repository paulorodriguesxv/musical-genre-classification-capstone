import wave
import pylab
import numpy as np
import pandas as pd
import librosa
import sklearn
import os
from genre import Genre
from helpers.files import get_genre

MFCC_BANDS = 13
FFT_WIN = 1024
SAMP_RATE = 22050
OVERLAP = 0.4
HOP = int(np.ceil((1-OVERLAP)*FFT_WIN))

def open_tune(path_to_tune, duration=-1):
    tune = wave.open(path_to_tune, 'r')
    if duration == -1:
        frames = tune.readframes(-1)
    else:
        frames = tune.readframes(duration*SAMP_RATE)
    freqs = pylab.fromstring(frames, np.int16)
    tune.close()

    return freqs

def extract_frequencies(tune, duration=None, normalized=False):
    tune = wave.open(tune, 'r')
    try:
        frames = tune.readframes(-1) if not duration else tune.readframes(duration*SAMP_RATE)

        freqs = pylab.fromstring(frames, np.int16) / 1.0

        # Normalize frequencies
        if normalized:
            freqs = freqs / float(2 ** 15)
    finally:
        tune.close()

    return freqs

def get_feature_names():
    return [x.name.lower() for x in Genre]

def populate_dataframe(main_directory, duration = 60):
    colnames = ['genre', 'frequence', 'zcr', 'rms', 'sc']

    for i in range(MFCC_BANDS):
        colnames.append(f'mfcc{i}')
        colnames.append(f'mfccd{i}')


    df = pd.DataFrame(columns=colnames)
    
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.wav'):
                feats = []

                path_to_tune = root + '/' + file
                genre = get_genre(path_to_tune)
                feats.append(genre)

                frequence = extract_frequencies(path_to_tune, duration=duration)

                zcr = librosa.feature.zero_crossing_rate(frequence)[0]
                rms = librosa.feature.rmse(frequence)[0]
                sc = librosa.feature.spectral_centroid(frequence, sr=SAMP_RATE)[0]
                mfcc = librosa.feature.mfcc(frequence, sr=SAMP_RATE, n_mfcc=MFCC_BANDS, n_fft=FFT_WIN, hop_length=HOP)
                mfccd = librosa.feature.delta(mfcc)

                feats.append(frequence)
                feats.append(zcr)
                feats.append(rms)
                feats.append(sc)

                for i in range(MFCC_BANDS):
                    feats.append(mfcc[i])
                    feats.append(mfccd[i])

                df.loc[len(df)] = feats

    return df

def prepare_data(df):
    df_f = pd.DataFrame()

    df_f['genre'] = df['genre'].astype('category')

    df_f['ZCR mean'] = [np.mean(x.reshape(1, -1)) for x in df['zcr']]
    df_f['ZCR std'] = [np.std(x.reshape(1, -1)) for x in df['zcr']]

    df_f['RMS mean'] = [np.mean(x.reshape(1, -1)) for x in df['rms']]
    df_f['RMS std'] = [np.std(x.reshape(1, -1)) for x in df['rms']]

    df_f['SC mean'] = [np.mean(x.reshape(1, -1)) for x in df['sc']]
    df_f['SC std'] = [np.std(x.reshape(1, -1)) for x in df['sc']]

    for i in range(1, MFCC_BANDS):    
        df_f['MFCC' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in df[f'mfcc{i}']]
        df_f['MFCC' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in df[f'mfcc{i}']]
        df_f['MFCCD' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in df[f'mfccd{i}']]
        df_f['MFCCD' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in df[f'mfccd{i}']]
    
    cat_columns = df.select_dtypes(['category']).columns
    df_f[cat_columns] = df[cat_columns].apply(lambda l: l.cat.codes)

    features = list(df_f.columns[1:])
    labels = df_f.columns[0]
    X = df_f[features]
    y = df_f[labels]

    return features, labels, X, y