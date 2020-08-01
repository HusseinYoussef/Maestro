import os
import argparse
import shutil
import stempeg
import soundfile as sf
from tqdm import tqdm
from glob import glob
import numpy as np
import math
from featurizer import STFT, Spectrogram


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
    
def construct_splitted_dataset(original_data_path, # the directory of the tracks.
                                x_path, y_path, # the two directories for the outputs.
                                stem_type, # stem_type => ['drums','bass', 'other', 'vocals' ]
                                seconds, # number of seconds which be in the output tracks.
                                frame_length= 4096, frame_step= 1024, sample_rate= 44100,freq_bands= 2049, channels= 2):
    ''' 
        this function splits the tracks which located in original_data_path,
        then construct two directories in x_path & y_path and locates the output on them.
        the output will be two folders of matrices [spectrugrams] 
    '''
    check_existance = lambda path: os.mkdir(path) if not(os.path.exists(path)) else None

    check_existance(x_path) if x_path is not None else print('[WARNING] no path given for X')
    check_existance(y_path) if y_path is not None else print('[WARNING] no path given for Y')

    if stem_type not in ['drums','bass', 'other', 'vocals']:
        return NameError(f'{stem_type} is not exist in our data')
    
    calc_frames = lambda samples: ((samples - frame_length) // frame_step + 1)
    frames = calc_frames(seconds * sample_rate)

    def expand_track_for_frames(track):
        wanted_frames = int(math.ceil(calc_frames(len(track)) / frames )) * frames
        wanted_len = (wanted_frames - 1) * frame_step + frame_length
        assert(calc_frames(wanted_len) == wanted_frames)
        padding = wanted_len - len(track)
        track = np.append(track, track[0:padding,:], axis= 0)
        assert(len(track) == wanted_len)
        return track

    label = 0
    def transform_and_save(path, stem, out_path):
        track, sample_rate = sf.read(f'{path}/{stem}.wav', always_2d=True)
        track = expand_track_for_frames(track).T
        spec = np.transpose((Spectrogram())((STFT())(track[None,...]))[0],(2,1,0))
        spec = np.reshape(spec, ( spec.shape[0] // frames, frames, spec.shape[1], spec.shape[2]) )
        for i in range(spec.shape[0]):
            np.save(f'{out_path}/{label}{i}.npy', spec[i])

    tracks = glob(f'{original_data_path}/*')
    for track_path in tqdm(tracks, total=len(tracks)):

        if x_path is not None: transform_and_save(track_path, 'mixture', x_path)
        if y_path is not None: transform_and_save(track_path, stem_type, y_path)
        
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