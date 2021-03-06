import numpy as np
import argparse
import random
import os
import torch
from glob import glob
from utils import audio_info, audio_loader
from tqdm import tqdm

class Transform:

    def __init__(self, functions:list):
        self.functions = functions

    def __call__(self, audio):
        for fn in self.functions:
            audio = fn(audio)
        return audio


def gain(audio, low=0.25, high=1.25):
    """Apply random gain to the signal between low and high"""
    g = low + np.random.rand() * (high-low)
    return audio * g

def channel_swap(audio):
    """Swap channels of stereo audio with a probabilty of 0.5"""
    if audio.shape[0] == 2 and np.random.rand() < 0.5:
        audio[[0,1]] = audio[[1,0]]
    return audio


def load_dataset(root, target='vocals', seq_duration=5, seed=42, linear_mix=True, augmentation=['gain', 'channel_swap']):
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
        seq_duration=seq_duration,
        samples_per_track=64,
        transformation=aug,
        seed=seed
    )
    valid_dataset = Dataset(
        root=root,
        split='valid',
        target=target,
        samples_per_track=1,
        linear_mix=linear_mix,
        seq_duration=None
    )

    return train_dataset, valid_dataset

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root:str,
        split='train',
        target='vocals',
        samplerate=44100,
        transformation=lambda audio: audio,
        samples_per_track=64,
        seq_duration=5,
        random_chunk=False,
        random_mix=False,
        linear_mix=True,
        seed=42
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.target = target
        self.samplerate = samplerate
        self.sources = ['vocals', 'drums', 'bass', 'other']
        self.transformation = transformation
        self.samples_per_track = samples_per_track
        self.seq_duration = seq_duration
        self.random_chunk = random_chunk
        self.random_mix = random_mix
        self.linear_mix = linear_mix
        self.tracks = self.get_tracks()
        self.generator = range(len(self.tracks))
        self.target_index = self.sources.index(self.target)
        random.seed(seed)

    def __getitem__(self, index) -> torch.tensor: 
        
        # breakpoint()
        track = self.tracks[index // self.samples_per_track]
        if self.split == 'train' or self.linear_mix:

            track_sources = []
            for source in self.sources:
                
                if self.random_mix:
                    rand_index = random.choice(self.generator)
                    track = self.tracks[rand_index]
                if self.random_chunk and self.seq_duration:
                    start = random.uniform(0, track['duration'] - self.seq_duration)
                else:
                    start = 0

                source_audio, _ = audio_loader(f'{track["path"]}/{source}.wav', start=start, dur=self.seq_duration)

                source_audio = self.transformation(source_audio)
                track_sources.append(torch.from_numpy(source_audio).float())

            stems = torch.stack(track_sources, dim=0)
            
            # linear mix
            x = stems.sum(0)
            # Target stem
            y = stems[self.target_index]
        else:

            #Load non-linear mixture
            mix_path = f'{track["path"]}/mixture.wav'
            x, _ = audio_loader(mix_path, start=0, dur=self.seq_duration)
            x = torch.from_numpy(x).float()

            target_path = f'{track["path"]}/{self.target}.wav'
            y, _ = audio_loader(target_path, start=0, dur=self.seq_duration)
            y = torch.from_numpy(y).float()

        return x, y

    def __len__(self):
        return len(self.tracks) * self.samples_per_track
    
    def __str__(self):
        print(f'Target = {self.target}')
        print(f'samples per track = {self.samples_per_track}')
        print(f'seq_duration = {self.seq_duration}')
        print(f'Length = {len(self)}')
        return ""
        
    def get_tracks(self) -> list:

        path = f'{self.root}/{self.split}/*'
        dirs = glob(path)

        tracks = []
        # breakpoint()
        for track_path in tqdm(dirs, desc=f'{self.split}_tracks', total=len(dirs)):
            fname = os.path.basename(track_path)

            # Make sure the track has the target source
            target_path = f'{track_path}/{self.target}.wav'
            if not os.path.exists(target_path):
                continue

            info = audio_info(target_path)
            if (self.seq_duration and info['duration'] >= self.seq_duration) or self.seq_duration is None:
                tracks.append({
                    'path': track_path,
                    'name': os.path.basename(track_path),
                    **info
                })

        return tracks

    def track_paths(self):
        """Print names of the tracks"""

        for track in self.tracks:
            print(track['name'])
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--target', type=str, default='vocals',
                        help='target source')

    parser.add_argument('--root', type=str, required=True, 
                            help='root to dataset directory')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')
                        
    parser.add_argument('--batch-size', type=int, default=16,
                            help='number of batches')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers for dataloader.')
                        
    # Model Parameters
    parser.add_argument('--seq-duration', type=int, default=5,
                            help='duration of samples')
    args, _ = parser.parse_known_args()
    
    train_dataset, valid_dataset = load_dataset(
        root=args.root,
        target=args.target,
        seq_duration=args.seq_duration,
        seed=42,
        linear_mix=False
    )

    total_training_duration = 0
    # for k in tqdm(range(len(train_dataset))):
    #     x, y = train_dataset[k]
    #     total_training_duration += x.shape[1] / train_dataset.samplerate

    torch.manual_seed(42)
    random.seed(42)
    use_cuda = torch.cuda.is_available()
    dataloader_params = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_params)
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    from model import UMX
    mymodel = UMX(
        frame_length=4096,
        frame_step=1024,
        rate=train_dataset.samplerate,
        n_channels=2,
        n_hidden=512,
        input_mean=None,
        input_scale=None,
        max_freq=16000,
        fc_bias=True,
        output_norm=False,
        n_layer=3
    )

    # # paths = train_dataset.track_paths()
    for x, y in tqdm(valid_sampler):
        pass
    for x, y in tqdm(train_sampler):
        pass
