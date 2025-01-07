import pandas as pd
import numpy as np

df = pd.read_csv('./data_path/annotation_.csv')
video_names = df['video_name'].tolist()
np.random.shuffle(video_names)

train_split = int(0.7 * len(video_names))
val_split = int(0.1 * len(video_names)) + train_split

train_videos = video_names[:train_split]
val_videos = video_names[train_split:val_split]
test_videos = video_names[val_split:]

def remove_extension(video_name):
    return ".".join(video_name.split(".")[:-1])

with open('./data/train_list.txt', 'w') as f:
    for video in train_videos:
        f.write(f"{remove_extension(video)}\n")

with open('./data/val_list.txt', 'w') as f:
    for video in val_videos:
        f.write(f"{remove_extension(video)}\n")

with open('./data/test_list.txt', 'w') as f:
    for video in test_videos:
        f.write(f"{remove_extension(video)}\n")