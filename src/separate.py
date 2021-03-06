import argparse
import torch
import os
import scipy
import json
import numpy as np
from model import UMX
from tqdm import tqdm
import soundfile as sf
import reconstruction

def load_model(target, path, name, device='cpu'):
    
    model_path = os.path.join(path, target+f'_{name}.pth')
    if not os.path.exists(model_path):
        raise NameError('Model does not exist')

    with open(os.path.join(path, target+f'_{name}_params.json'), 'r') as fin:
        params = json.load(fin)

    # breakpoint()
    model_state = torch.load(model_path, map_location=device)
    
    # breakpoint()
    model = UMX(
        frame_length=params['args']['frame_length'],
        frame_step=params['args']['frame_step'],
        n_channels=params['args']['channels'],
        n_hidden=params['args']['hidden_size'],
        max_freq=params['args']['bandwidth'],
        output_norm=params['args']['output_norm'],
        rate=params['sample_rate']
    )

    # Load model parameters
    model.load_state_dict(model_state)
    # Use center in stft for better reconstruction
    model.stft.center = True
    # evaluation mode
    model.eval()
    model.to(device)

    return model

def separate(
    audio,
    targets,
    models_path,
    model_name,
    niter=1,
    residual_model=False,
    device='cpu'
):

    # audio shape: samples, channels
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
     
    # audio shape: batch_size, channels, samples
    source_names = []
    models_out = []

    for j, target in enumerate(tqdm(targets, desc='Models')):

        target_model = load_model(target=target, path=models_path, name=model_name, device=device)
        # breakpoint()
        feats = target_model.extract_features(audio_torch)
        output = target_model(feats).cpu().detach().numpy()
        
        # output shape is: (batch_size, frames, channels, freq_bins)
        models_out.append(output[0, ...])
        source_names += [target]

    # shapes should be: (frames, freq_bins, channels) for wiener filter

    models_out = np.transpose(np.array(models_out), (1, 3, 2, 0))
    # models_out shape is: (frames, freq_bins, channels, n_sources)

    audio_stft = target_model.stft(audio_torch)

    # breakpoint()
    audio_stft = audio_stft[0].transpose(2, 1, 0)
    # audio_stft shape is: (frames, freq_bins, channels)

    estimates = reconstruction.reconstruct(
                                mag_estimates=models_out,
                                mix_stft=audio_stft,
                                targets=targets,
                                niter=niter,
                                frame_length=target_model.stft.frame_length,
                                frame_step=target_model.stft.frame_step,
                                residual=residual_model
                            )

    return estimates


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--path', type=str, required=True,
                        help='path to audio file')

    parser.add_argument('--model', type=str, required=True,
                        help='path to models directory')
    
    parser.add_argument('--targets', nargs='+', default=['vocals', 'drums', 'bass', 'other'],
                        help='targets to output')

    parser.add_argument('--name', type=str, default='try5',
                        help='model name.')

    parser.add_argument('--iter', type=int, default=1,
                        help='number of iterations for wiener filter.')

    parser.add_argument('--start', type=int, default=0,
                        help='start of audio')

    parser.add_argument('--dur', type=int, default=None,
                        help='duration of the audio')

    parser.add_argument('--residual', action='store_true', 
                        help='compute residual target')

    args, _ = parser.parse_known_args()

    rate = 44100
    if args.dur is not None:
        audio_file, rate = sf.read(args.path, start=args.start*rate, stop=(args.start*rate + int(args.dur * rate)))
    else:
        audio_file, rate = sf.read(args.path, start=args.start*rate, stop=None)

    estimates = separate(audio=audio_file, targets=args.targets, niter=args.iter, 
                    models_path=args.model, model_name=args.name, residual_model=args.residual)
    
    for target in tqdm(args.targets, desc='Writing Outputs'):
        if target in estimates:
            sf.write(f'{target}.wav', estimates[target], rate)
        else:
            print(f'Target {target} not found.')
