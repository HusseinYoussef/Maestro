import numpy as np
import argparse
import random
import musdb
import os
import torch
from glob import glob
from utils import audio_info, audio_loader

class Transform:

    def __init__(self, functions:list):
        self.functions = functions

    def __call__(self, audio):
        for fn in self.functions:
            audio = fn(audio)
        return audio


def gain(self, audio, low=0.25, high=1.25):
    """Apply random gain to the signal between low and high"""
    g = low + np.random.rand() * (high-low)
    return audio * g

def channel_swap(self, audio):
    """Swap channels of stereo audio with a probabilty of 0.5"""
    if audio.shape[0] == 2 and np.random.rand() >= 0.5:
        audio[[0,1]] = audio[[1,0]]
    else:
        return audio


def load_dataset(root, target='vocals', seq_duration=5, augmentation=['gain', 'channel_swap']):
    """Load Train and Validation datasets"""

    if augmentation is None:
        augmentation = []

    aug = Transform([globals()[x] for x in augmentation])

    train_dataset = Dataset(
        root=root,
        split='train',
        target=target,
        random_chunk=True,
        random_mix=True,
        samples_per_track=64,
        transformation=aug
    )
    valid_dataset = Dataset(
        root=root,
        split='valid',
        target=target,
        samples_per_track=1,
    )

    return train_dataset, valid_dataset

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root:str,
        split='train',
        target='vocals',
        transformation=lambda audio: audio,
        samples_per_track=1,
        seq_duration=5,
        random_chunk=False,
        random_mix=False
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.target = target
        self.sources = ['vocals', 'drums', 'bass', 'other']
        self.transformation = transformation
        self.samples_per_track = samples_per_track
        self.seq_duration = seq_duration
        self.random_chunk = random_chunk
        self.random_mix = random_mix
        self.tracks = self.get_tracks()

    def __getitem__(self, index):
        
        track_sources = []
        track = self.tracks[index // self.samples_per_track]

        if self.random_chunk and self.seq_duration:
            start = random.uniform(0, track['duration'] - self.seq_duration)
        else:
            start = 0

        target_path = f'{track["path"]}/{self.target}.wav'
        target_audio, _ = audio_loader(target_path, start=start, dur=self.seq_duration)

        target_audio = self.transformation(target_audio)
        track_sources.append(torch.Tensor(target_audio))

        for source in self.sources:
            if source != self.target:

                if self.random_mix:
                    random_index = random.choice(range(len(self.tracks)))
                    interferer_track = self.tracks[random_index]
                else:
                    interferer_track = track

                if self.random_chunk and self.seq_duration:
                    start = random.uniform(0, interferer_track['duration'] - self.seq_duration)
                
                source_path = f'{interferer_track["path"]}\{source}.wav'
                if not os.path.exists(source_path):
                    continue

                source_audio, _ = audio_loader(source_path, start=start, dur=self.seq_duration)
                source_audio = self.transformation(source_audio)

                track_sources.append(torch.Tensor(source_audio))

        stems = torch.stack(track_sources)
        
        # linear mix
        x = stems.sum(0)
        # Target stem
        y = stems[0]

        return x, y

    def __len__(self):
        return len(self.tracks)
    
    def __str__(self):
        print(f'Target = {self.target}')
        print(f'samples per track = {self.samples_per_track}')
        print(f'Length = {len(self.tracks)}')
        return ""
        
    def get_tracks(self):

        path = f'{self.root}/{self.split}/*'
        dirs = glob(path)

        tracks = []
        # breakpoint()
        for track_path in dirs:
            fname = os.path.basename(track_path)

            target_path = f'{track_path}/{self.target}.wav'
            if not os.path.exists(target_path):
                continue

            info = audio_info(target_path)
            if (self.seq_duration and info['duration'] >= self.seq_duration) or self.seq_duration is None:
                tracks.append({
                    'path':track_path,
                    **info
                })

        return tracks


if __name__ == "__main__":

    # sum = 0
    # c = 0
    # mus = musdb.DB(root="F:/CMP 2020/GP/musdb18", split="valid", subsets='train')
    # lis = []
    # for track in mus:
        
    #     lis.append(track.duration)
    #     c += 1
    #     sum += track.duration

    dataset = Dataset(root='F:/CMP 2020/GP/Datasets/Maestro', target='drums')
    # breakpoint()
    # dataset.get_tracks()