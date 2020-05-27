import os
import argparse
import shutil
import stempeg
import soundfile as sf
from tqdm import tqdm
from glob import glob
import numpy as np


def musdb_parser(args):

    root_path = args.root
    dest_path = args.dest

    validation_tracks = [
        'Actions - One Minute Smile',
        'Clara Berry And Wooldog - Waltz For My Victims',
        'Johnny Lokke - Promises & Lies',
        'Patrick Talbot - A Reason To Leave',
        'Triviul - Angelsaint',
        'Alexander Ross - Goodbye Bolero',
        'Fergessen - Nos Palpitants',
        'Leaf - Summerghost',
        'Skelpolu - Human Mistakes',
        'Young Griffo - Pennies',
        'ANiMAL - Rockshow',
        'James May - On The Line',
        'Meaxic - Take A Step',
        'Traffic Experiment - Sirens'
    ]

    if not(os.path.exists(f'{dest_path}/musdb18')):

        os.mkdir(f'{dest_path}/musdb18')
        dest_train_path = dest_path + '/musdb18/train'
        dest_test_path = dest_path + '/musdb18/test'

        stems = ['mixture', 'drums', 'bass', 'other', 'vocals']

        for split_path, split in tqdm(zip([dest_train_path, dest_test_path], ['train', 'test'])):
            with open(f'{dest_path}/musdb18/{split}.csv', 'w') as fo:

                fo.write('fname,duration\n')

                if not(os.path.exists(split_path)):
                    
                    os.mkdir(split_path)
                    # print(f'\nProcessing {split} files')
                    files = glob(f'{root_path}/{split}/*.mp4')
                    
                    if args.n:
                        n_tracks = min(len(files), args.n)
                    else:
                        n_tracks = len(files)

                    for track_path in tqdm(files[:n_tracks], total=n_tracks):

                        filename = os.path.basename(track_path)
                        
                        # remove extension
                        fname = filename.split('.stem')[0]
                        if fname not in validation_tracks:
                            track_dir = f'{split_path}/{fname}'
                        else:
                            if not os.path.exists(f'{dest_path}/musdb18/valid'):
                                os.mkdir(f'{dest_path}/musdb18/valid')
                            track_dir = f'{dest_path}/musdb18/valid/{fname}'

                        if not(os.path.exists(track_dir)):
                            os.mkdir(track_dir)

                        signal, rate = stempeg.read_stems(track_path)
                        duration = signal.shape[1] // rate

                        # breakpoint()
                        for idx, stem in enumerate(stems):
                            sf.write(f'{track_dir}/{stem}.wav', signal[idx], rate)
                        fo.write(f'{fname},{duration}\n')

def ccmixter_parser(args):

    root_path = args.root
    dest_path = args.dest

    if not(os.path.exists(f'{dest_path}/ccmixter')):
        os.mkdir(f'{dest_path}/ccmixter')
        os.mkdir(f'{dest_path}/ccmixter/train')

        dirs = glob(root_path+'/*')
        stems = ['mixture', 'other', 'vocals']
                    
        with open(f'{dest_path}/ccmixter/train.csv', 'w') as fo:
            
            fo.write('fname,duration\n')
            if args.n:
                n_tracks = min(len(dirs), args.n)
            else:
                n_tracks = len(dirs)

            for directory in tqdm(dirs[:n_tracks], total=n_tracks):

                if not os.path.isdir(directory):
                    continue
                
                fname = os.path.basename(directory)
                track_dir = f'{dest_path}/ccmixter/train/{fname}'
                os.mkdir(track_dir)

                files = glob(directory+'/*.wav')

                # breakpoint()
                for idx, stem in enumerate(stems):
                    signal, rate = sf.read(files[idx])
                    duration = signal.shape[0]//rate
                    shutil.copy(files[idx], f'{track_dir}/{stem}.wav')
                fo.write(f'{fname},{duration}\n')

def HHDS_parser(args):

    root_path = args.root
    dest_path = args.dest

    duplicates = [
        '001 - ANiMAL - Clinic A',
        '002 - ANiMAL - Rockshow',
        '052 - ANiMAL - Easy Tiger'
    ]

    if not os.path.exists(f'{dest_path}/HHDS'):
        os.mkdir(f'{dest_path}/HHDS')
        os.mkdir(f'{dest_path}/HHDS/train')
        os.mkdir(f'{dest_path}/HHDS/valid')

        for split, dest_split in zip(['Dev', 'Test'], ['train', 'valid']):
            for folder in ['Mixtures', 'Sources']:
                
                dirs = glob(f'{root_path}/{folder}/{split}/*')
                for track_dir in tqdm(dirs, desc=f'{folder} files'):

                    fname = os.path.basename(track_dir)
                    if fname in duplicates:
                        continue
                    if not os.path.exists(f'{dest_path}/HHDS/{dest_split}/{fname}'):
                        os.mkdir(f'{dest_path}/HHDS/{dest_split}/{fname}')
                    
                    files = glob(f'{track_dir}/*.wav')
                    for track in files:
                        shutil.copy(track, f'{dest_path}/HHDS/{dest_split}/{fname}')

def sampling(root_path, dest_path, datasets, seq_duration=None):

    if not os.path.exists(f'{dest_path}/Maestro'):
        os.mkdir(f'{dest_path}/Maestro')
        os.mkdir(f'{dest_path}/Maestro/train')
        os.mkdir(f'{dest_path}/Maestro/valid')

    # datasets = ['musdb18', 'ccmixter', 'HHDS']
    sources = ['vocals', 'drums', 'bass', 'other', 'mixture']
    # breakpoint()
    for split in ['train', 'valid']:
        for dataset in datasets:

            split_dir = f'{root_path}/{dataset}/{split}'
            if not os.path.exists(split_dir):
                continue

            tracks = glob(f'{split_dir}/*')
            for track_path in tqdm(tracks, desc=f'{dataset} {split} files', total=len(tracks)):
                fname = os.path.basename(track_path)

                for source in sources:
                    # Check if the track has this source
                    source_path = f'{track_path}/{source}.wav'
                    if not os.path.exists(source_path):
                        continue
                    signal, rate = sf.read(source_path)
                    
                    if seq_duration is not None:
                        interval = rate * seq_duration
                    else:
                        interval = signal.shape[0]
                        
                    for i in range(0, signal.shape[0]-interval + 1, interval):
                        if seq_duration is not None:
                            sample = signal[i:i+interval, :]
                            save_dir = f'{dest_path}/Maestro/{split}/{fname}_{i//interval + 1}'
                        else:
                            sample = signal
                            save_dir = f'{dest_path}/Maestro/{split}/{fname}'
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        sf.write(f'{save_dir}/{source}.wav', sample, rate)

def expand_track(track, divisor): # track.shape = (samples, channel)
    ''' this function expand the track to be divisable by divisor '''
    
    ''' Logic of the function: 
    Take values from the begining of the track and append them to the end of the track to make
    the final size of the track is divisble by divisor. '''

    siz = len(track) # number of samples.
    new_siz = (siz//divisor + 1) * divisor
    reminder = int((new_siz-siz) % divisor)
    return np.append(track,track[0:reminder,:],axis = 0)

def split_track(track, sample_rate, t, write= False, folder_path=None, track_name=None):
    ''' this function split the track to x tracks each of them is t seconds and save them in folder_path with track_name(0 -> x)'''
    n = sample_rate * t # convert seconds to samples.
    track = expand_track(track, n)
    tracks = len(track) // n
    splitted = []
    for i in range(tracks):
        start = i*n
        end = (i+1)*n
        if write:
            file_path = folder_path + '/' + track_name + str(i) + ".wav"
            sf.write(file_path, track[start:end,:], sample_rate)
        else:
            splitted.append(track[start:end,:])

    if not write:
        return splitted
    
def construct_splitted_dataset(original_data_path, x_path, y_path, stem, seconds):
  ''' this function splits the tracks which located in original_data_path, and then construct two directories in x_path & y_path and locates the output on them. '''
  label = 0

  if not(os.path.exists(x_path)):
    os.mkdir(x_path)

  if not(os.path.exists(y_path)):
    os.mkdir(y_path)

  original_data_path += '/*'
  tracks = glob(original_data_path)

  for track_path in tqdm(tracks, total=len(tracks)):
    mix_path = track_path + '/mixture.wav'
    stem_path = track_path + f'/{stem}.wav'

    track, sample_rate = sf.read(mix_path, always_2d=True)
    split_track(track, sample_rate, seconds, x_path, str(label), write= True)

    track, sample_rate = sf.read(stem_path, always_2d=True)
    split_track(track, sample_rate, seconds, y_path, str(label), write= True)
    label+=1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare Dataset')

    parser.add_argument(
        '--name', type=str, required=True, default='musdb18', help='dataset name'
    )

    parser.add_argument(
        '--n', type=int, default=3, help='number of tracks of each dataset, 0 for all tracks'
    )

    parser.add_argument(
        '--root', type=str, required=True ,help='root path of dataset'
    )

    parser.add_argument(
        '--dest', type=str, required=True, help='destination path of dataset'
    )
    parser.add_argument(
        '--seq-duration', type=int, help='length of sample'
    )

    parser.add_argument('--datasets', nargs='+', help='datasets'
    )

    args, _ = parser.parse_known_args()

    if args.name == 'musdb18':
        musdb_parser(args)
    elif args.name == 'ccmixter':
        ccmixter_parser(args)
    elif args.name == 'HHDS':
        HHDS_parser(args)
    elif args.name == 'maestro':
        sampling(args.root, args.dest, args.datasets, args.seq_duration)