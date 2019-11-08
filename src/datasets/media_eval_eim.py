import os
import librosa
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset


class MediaEvalEiMDataset(Dataset):

    def __init__(self,
            audio_dir,
            arousal_average_path,
            valence_average_path):
        self.audio_path_list = glob.glob(os.path.join(audio_dir, '*'))
        self.df_arousal_average = pd.read_csv(arousal_average_path)
        self.df_valence_average = pd.read_csv(valence_average_path)

    def __getitem__(self, idx):

        song_path = self.audio_path_list[idx]
        song_id = int(os.path.basename(os.path.splitext(song_path)[0]))
        arousal_average = self.df_arousal_average[self.df_arousal_average['song_id'] == song_id].values[:, 1:]
        valence_average = self.df_valence_average[self.df_valence_average['song_id'] == song_id].values[:, 1:]

        y, sr = librosa.load(song_path)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel = torch.Tensor(mel)

        return mel, arousal_average, valence_average

    def __len__(self):
        return len(self.audio_path_list)
