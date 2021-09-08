import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/Ahmed/Desktop/dissertation/archive/recordings/recordings_wave/wave.csv')
labels = pd.read_csv("C:/Users/Ahmed/Desktop/dissertation/archive/labels_5.csv") 

import numpy as np
def injection(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data



def shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


import librosa
def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


injections = []
injections_labels = []
for i in range(0,len(df)):
    wav = injection(np.array(df.iloc[i]),0.05)
    injections.append(wav[:40000])
    
    
shifts = []
shifts_labels = []
for i in range(0,len(df)):
    wav = shift(np.array(df.iloc[i]),  22050, 0.05, 'right')
    shifts.append(wav[:40000])
    
    
speeds = []
speeds_labels = []
for i in range(0,len(df)):
    wav = speed(np.array(df.iloc[i]), 0.85)
    speeds.append(wav[:40000])
    
    
pieces = (pd.DataFrame(np.array(speeds)), pd.DataFrame(np.array(shifts)),pd.DataFrame(np.array(injections)))
df_aug = pd.concat(pieces, ignore_index = True)

pieces1 = [df.iloc[:,1:],df_aug]
df_full = pd.concat(pieces1)


labels_pieces = (labels,labels,labels,labels)
labels_aug = pd.concat(labels_pieces)



### aug computational expensive 
### data leakage issues -> augumented data would be kept in just training or just test 
### diss aims to evaulate archtertures 

## -> compare speed of training , ac of testing etc based on low data 