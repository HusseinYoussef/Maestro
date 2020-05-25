import argparse
import torch
import os
import scipy
import norbert
import json
import numpy as np
from model import UMX
from tqdm import tqdm
import soundfile as sf
from featurizer import iSTFT

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

def istft(X, frame_length=4096, frame_step=1024):
    
    istft_obj = iSTFT(frame_length=frame_length, frame_step=frame_step)
    audio = istft_obj(X / (frame_length / 2), boundary=True)

    return audio

def separate(
    audio,
    targets,
    models_path,
    model_name,
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu'
):

    # audio shape: samples, channels
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
     
    # audio shape: batch_size, channels, samples
    source_names = []
    models_out = []

    for j, target in enumerate(tqdm(targets, desc='Targets')):

        target_model = load_model(target=target, path=models_path, name=model_name, device=device)
        # breakpoint()
        feats = target_model.extract_features(audio_torch)
        output = target_model(feats).cpu().detach().numpy()
        
        if softmask:
            # only exponentiate the model if we use softmask
            output = output**alpha

        # output shape is: (batch_size, frames, channels, freq_bins)
        models_out.append(output[0, ...])
        source_names += [target]

    # shapes should be: (frames, freq_bins, channels) for wiener filter

    models_out = np.transpose(np.array(models_out), (1, 3, 2, 0))
    # models_out shape is: (frames, freq_bins, channels, n_targets)

    audio_stft = target_model.stft(audio_torch)

    # breakpoint()
    audio_stft = audio_stft[0].transpose(2, 1, 0)
    # audio_stft shape is: (frames, freq_bins, channels)

    if residual_model or len(targets) == 1:
        models_out = norbert.residual_model(models_out, audio_stft, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(models_out, audio_stft.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            frame_length=target_model.stft.frame_length,
            frame_step=target_model.stft.frame_step
        )
        estimates[name] = audio_hat.T
        # estimate shape: samples, channels

    return estimates


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--path', type=str, required=True,
                        help='path to audio file')

    parser.add_argument('--model', type=str, required=True,
                        help='path to models directory')
    
    parser.add_argument('--targets', nargs='+', default=['vocals'],
                        help='targets to output')

    parser.add_argument('--name', type=str, default='try5',
                        help='model name.')

    parser.add_argument('--iter', type=int, default=1,
                        help='number of iterations for wiener filter.')

    parser.add_argument('--start', type=int, default=0,
                        help='start of audio')

    parser.add_argument('--dur', type=int, default=None,
                        help='duration of the audio')
    
    args, _ = parser.parse_known_args()

    rate = 44100
    if args.dur is not None:
        audio_file, rate = sf.read(args.path, start=args.start*rate, stop=(args.start*rate + int(args.dur * rate)))
    else:
        audio_file, rate = sf.read(args.path, start=args.start*rate, stop=None)

    estimates = separate(audio=audio_file, targets=args.targets, niter=args.iter, 
                    models_path=args.model, model_name=args.name)
    
    for target in tqdm(args.targets, desc='Writing Outputs'):
        if target in estimates:
            sf.write(f'{target}.wav', estimates[target], rate)
        else:
            print(f'Target {target} not found.')
