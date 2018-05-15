import os
from helpers import files, plots
from genre import Genre
from feature import SoundFeature
from helpers import npdata
import librosa
import matplotlib.pyplot as plt
import numpy as np
import wave
import pandas as pd

waveform = []
for index, g in enumerate(Genre):
    wf = os.path.join(files.music_path(g.name.lower()), str(index+1) + ".wav")
    waveform.append(wf)
    
#plots.plot_wavesform(waveform, n_seconds=15)
#plots.plot_superimprose_wave(waveform, SoundFeature.ZCR,15)



genres = ['blues', 'classical', 'jazz', 'pop', 'rock']
zcrs = []
"""
for tune in waveform:
    print("Opening: " + tune)
    freqs = npdata.extract_frequencies(tune, normalized=False)
    print("Extracting ZCR")
    zeros = librosa.feature.zero_crossing_rate(freqs)
    zcrs.append(zeros[0])
#
plots.box_plots(zcrs, genres, 'ZCR')

"""

"""
main_dir = files.music_path("")
df = npdata.populate_df(main_dir, duration=20)
plots.iqr_plots(df, genres, SoundFeature.FREQUENCE)

"""

#plots.plot_spectrograms(waveform, duration=20)

"""
main_dir = files.music_path("")
df = npdata.populate_df(main_dir, duration=20)
df = df.sort_values(by=["genre"])

matrix_data = (df.raw.apply(pd.Series).as_matrix())

x2=np.percentile(matrix_data,75,axis=1)-np.percentile(matrix_data,25,axis=1)

a=[]
for i in range(5):    
    a.append(x2[i*100:(i+1)*100])


color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

fig,ax=plt.subplots(figsize=(8,5))
ax.boxplot(a,labels=genres, sym='r+')
ax.set_title('inter-quartile range of amplitudes of audio signals in five genres')
fig.savefig('fig1.png', bbox_inches='tight')


"""

main_dir = files.music_path("")

df = npdata.populate_dataframe(main_dir, duration=15)

npdata.prepare_data(df)
