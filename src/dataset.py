import os
import argparse
import shutil
import stempeg
import soundfile as sf
from tqdm import tqdm
from glob import glob


def musdb_parser(root_path, dest_path):

    if not(os.path.exists(f'{dest_path}/musdb18')):

        root_train_path = root_path + '/train'
        root_test_path = root_path + '/test'

        dest_train_path = dest_path + '/musdb18/train'
        dest_test_path = dest_path + '/musdb18/test'

        stems = ['mixture', 'drums', 'bass', 'other', 'vocals']

        os.mkdir(f'{dest_path}/musdb18')
        with open(f'{dest_path}/musdb18/musdb.csv', 'w') as fo:

            fo.write('fname,duration,split\n')
            for split_path, split in tqdm(zip([dest_train_path, dest_test_path], ['train', 'test'])):

                if not(os.path.exists(split_path)):
                    
                    os.mkdir(split_path)
                    print(f'\nProcessing {split} files')
                    files = glob(root_train_path+'/*.mp4') if split == 'train' else glob(root_test_path+'/*.mp4')
                    
                    for track_path in tqdm(files, total=len(files)):

                        filename = os.path.basename(track_path)
                        
                        # remove extension
                        fname = filename.split('.')[0]
                        track_dir = f'{split_path}/{fname}'

                        if not(os.path.exists(track_dir)):
                            os.mkdir(track_dir)

                        signal, rate = stempeg.read_stems(track_path)
                        duration = signal.shape[1] // rate

                        # breakpoint()
                        for idx, stem in enumerate(stems):
                            sf.write(f'{track_dir}/{stem}.wav', signal[idx], rate)
                        fo.write(f'{fname},{duration},{split}\n')

def ccmixter_parser(root_path, dest_path):

    if not(os.path.exists(f'{dest_path}/ccmixter')):
        os.mkdir(f'{dest_path}/ccmixter')
        os.mkdir(f'{dest_path}/ccmixter/train')

        dirs = glob(root_path+'/*')
        stems = ['mixture', 'other', 'vocals']
                    
        with open(f'{dest_path}/ccmixter/ccmixter.csv', 'w') as fo:
            
            fo.write('fname,duration,split\n')
            for directory in tqdm(dirs, total=len(dirs)):

                fname = os.path.basename(directory)
                track_dir = f'{dest_path}/ccmixter/train/{fname}'
                os.mkdir(track_dir)

                files = glob(directory+'/*.wav')

                # breakpoint()
                for idx, stem in enumerate(stems):
                    signal, rate = sf.read(files[idx])
                    duration = signal.shape[0]//rate
                    shutil.move(files[idx], f'{track_dir}/{stem}.wav')
                fo.write(f'{fname},{duration},train\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare Dataset')

    parser.add_argument(
        '--name', type=str, required=True, default='musdb', help='dataset name'
    )

    parser.add_argument(
        '--root', type=str, required=True ,help='root path of dataset'
    )

    parser.add_argument(
        '--dest', type=str, required=True, help='destination path of dataset'
    )

    args, _ = parser.parse_known_args()

    if args.name == 'musdb':
        musdb_parser(args.root, args.dest)
    elif args.name == 'ccmixter':
        ccmixter_parser(args.root, args.dest)
    