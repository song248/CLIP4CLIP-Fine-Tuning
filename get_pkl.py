import pandas as pd
import pickle

csv_file_path = './data/annotation.csv'
pickle_file_path = 'raw-captions.pkl'

df = pd.read_csv(csv_file_path, usecols=['video_name', 'caption'])
df['video_name'] = df['video_name'].apply(lambda x: ".".join(x.split(".")[:-1]))

video_captions_dict = df.set_index('video_name')['caption'].to_dict()

with open(pickle_file_path, 'wb') as f:
    pickle.dump(video_captions_dict, f)
