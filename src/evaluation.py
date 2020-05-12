import argparse
import os
import museval
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

def eval(reference_dir, estimates_dir, output_dir=None, target='vocals'):

    scores = {}
    reference_glob = os.path.join(reference_dir, '*.wav')
    
    for reference_file in tqdm(glob(reference_glob), desc='reference_files'):

        track_name = os.path.basename(reference_file).split('.wav')[0]
        estimate_file = os.path.join(estimates_dir, f'{track_name}.wav')

        if not os.path.exists(estimate_file):
            raise ValueError('estimate file not found')

        reference = []
        estimates = []

        ref_audio, rate = sf.read(reference_file, always_2d=True)
        est_audio, rate = sf.read(estimate_file, always_2d=True)

        reference.append(ref_audio)
        estimates.append(est_audio)

        SDR, ISR, SIR, SAR = museval.evaluate(
                        reference,
                        estimates,
                    )
        m, s = divmod(ref_audio.shape[0]//rate, 60)
        scores[track_name] = {
            'duration': f'{m}:{s}',
            "SDR(dB)": np.nanmedian(SDR[0].tolist()),
            "Target": target
        }

    df = pd.DataFrame.from_dict(scores, orient='index')
    if output_dir is None:
        df.to_csv('results.csv')
    else:
        df.to_csv(f'{output_dir}/results.csv')

    return scores


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--ref', type=str,
                        help='path to reference files directory')
    parser.add_argument('--est', type=str,
                        help='path to estimates files directory')
    parser.add_argument('--output-dir', type=str,
                        help='path to directory to save results')

    args, _ = parser.parse_known_args()

    scores = eval(args.ref, args.est, args.output_dir)