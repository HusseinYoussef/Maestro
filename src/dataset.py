import os
import argparse
import shutil
import stempeg
import soundfile as sf
from tqdm import tqdm
from glob import glob


def musdb_parser(root_path, dest_path, seq_duration=None):

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
                    print(f'\nProcessing {split} files')
                    files = glob(f'{root_path}/{split}/*.mp4')
                    
                    for track_path in tqdm(files[:3], total=len(files)):

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

def ccmixter_parser(root_path, dest_path):

    if not(os.path.exists(f'{dest_path}/ccmixter')):
        os.mkdir(f'{dest_path}/ccmixter')
        os.mkdir(f'{dest_path}/ccmixter/train')

        dirs = glob(root_path+'/*')
        stems = ['mixture', 'other', 'vocals']
                    
        with open(f'{dest_path}/ccmixter/train.csv', 'w') as fo:
            
            fo.write('fname,duration\n')
            for directory in tqdm(dirs[:3], total=len(dirs)):

                fname = os.path.basename(directory)
                track_dir = f'{dest_path}/ccmixter/train/{fname}'
                os.mkdir(track_dir)

                files = glob(directory+'/*.wav')

                # breakpoint()
                for idx, stem in enumerate(stems):
                    signal, rate = sf.read(files[idx])
                    duration = signal.shape[0]//rate
                    shutil.move(files[idx], f'{track_dir}/{stem}.wav')
                fo.write(f'{fname},{duration}\n')


def sampling(root_path, seq_duration=None):

    if not os.path.exists(f'{root_path}/Maestro'):
        os.mkdir(f'{root_path}/Maestro')
        os.mkdir(f'{root_path}/Maestro/train')
        os.mkdir(f'{root_path}/Maestro/valid')

    datasets = ['musdb18', 'ccmixter']
    sources = ['vocals', 'drums', 'bass', 'other', 'mixture']
    breakpoint()
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
                            save_dir = f'{root_path}/Maestro/{split}/{fname}_{i//interval + 1}'
                        else:
                            sample = signal
                            save_dir = f'{root_path}/Maestro/{split}/{fname}'
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        sf.write(f'{save_dir}/{source}.wav', sample, rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare Dataset')

    parser.add_argument(
        '--name', type=str, required=True, default='musdb18', help='dataset name'
    )

    parser.add_argument(
        '--root', type=str, required=True ,help='root path of dataset'
    )

    parser.add_argument(
        '--dest', type=str, required=True, help='destination path of dataset'
    )

    args, _ = parser.parse_known_args()

    # if args.name == 'musdb18':
    #     musdb_parser(args.root, args.dest)
    # elif args.name == 'ccmixter':
    #     ccmixter_parser(args.root, args.dest)
    
    sampling('F:\CMP 2020\GP\Datasets')