def get_wav(path):
    '''
    Load wav file from disk and down-samples to RATE
    '''

    y, sr = librosa.load(directory1+'/'+path)
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=sr, scale=True),sr)

lis = []
directory1 = 'C:/Users/Ahmed/Desktop/dissertation/archive/recordings/recordings_wave'
for filename in os.listdir(directory1):
    if filename.endswith("mp3.wav"):
        step = get_wav(filename)[:40000]
        lis.append(step)
        
    pd.Dataframe(lis)
        
### Look into length of audio 
        
## 