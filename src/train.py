import os
import torch
from torchsummary import summary
import argparse
import numpy as np
import tqdm
import json
from data import load_dataset, Dataset
import copy
import random
import sklearn.preprocessing
from featurizer import STFT, Spectrogram
from utils import save_checkpoint
from model import UMX

def train(device, model, train_sampler, optimizer) -> float:
    """Train an Epoch and return the average loss"""
    # set training mode for all layers specially (batch_norm, dropout)
    model.train()
    tot_error = 0
    n_samples = 0
    losses_avg = 0
    for x, y in tqdm.tqdm(train_sampler, desc='Training Batch'):

        # move the the batch to CPU/GPU
        x, y = x.to(device), y.to(device)
        
        # Clear the grads of the optimizer since pytorch accumlates the grads
        optimizer.zero_grad()
        
        # Compute model output spectrogram
        Y_hat = model(x)
        # Actual spectrogram
        Y = model.extract_features(y)
        # Compute average MSE over the batch
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        # Compute the grads using backpropagation  d(loss)/dw
        loss.backward()
        # move one step of optimizer
        optimizer.step()
    
        # summation of batch samples losses by multiplying by batch_size
        tot_error += loss.item() * Y.shape[0]
        n_samples += Y.shape[0]
    
    losses_avg = tot_error / n_samples
    return losses_avg

def valid(device, model, valid_sampler):
    """Evaluate the model over the validation set and return average loss"""
    # set evaluation mode for all layers specially (batch_norm, dropout)
    model.eval()
    tot_error = 0
    n_samples = 0
    losses_avg = 0

    # stop autograd from tracking history on Tensors to free memory and speed up
    with torch.no_grad():
        for x, y in valid_sampler:

            x, y = x.to(device), y.to(device)
            
            Y_hat = model(x)
            # Actual spectrogram
            Y = model.extract_features(y)
            # Compute average MSE over the batch
            loss = torch.nn.functional.mse_loss(Y_hat, Y)

            tot_error += loss.item() * Y.shape[0]
            n_samples += Y.shape[0]

    losses_avg = tot_error / n_samples
    return losses_avg

def standard_scaler(args, dataset:Dataset):
    """Calculate statistics for the dataset using standard Scaler"""

    scaler = sklearn.preprocessing.StandardScaler()
    # spec = torch.nn.Sequential(
    #     STFT(frame_length=args.frame_length, frame_step=args.frame_step),
    #     Spectrogram(mono=True)
    # )
    stft_obj = STFT(frame_length=args.frame_length, frame_step=args.frame_step)
    spec_obj = Spectrogram(mono=True)

    dataset_cpy = copy.deepcopy(dataset)
    dataset_cpy.samples_per_track = 1
    dataset_cpy.transformation = lambda audio:audio
    dataset_cpy.random_chunk = False
    dataset_cpy.random_mix = False
    dataset_cpy.seq_duration = None
    
    for index in tqdm.tqdm(range(len(dataset_cpy)), desc='Compute dataset statistics'):
        x, y = dataset_cpy[index]

        # Add None to compensate batch_size
        X = spec_obj(stft_obj(x[None, ...]))
        # X = spec(x[None, ...])
        scaler.partial_fit(np.squeeze(X).T)

    # breakpoint()
    std = np.maximum(scaler.scale_, 1e-4*np.max(scaler.scale_))

    return scaler.mean_, std

def minmax_scaler(args, dataset:Dataset):
    """Calculate statistics for the dataset using MinMax Scaler"""

    dataset_cpy = copy.deepcopy(dataset)
    dataset_cpy.samples_per_track = 1
    dataset_cpy.transformation = lambda audio:audio
    dataset_cpy.random_chunk = False
    dataset_cpy.random_mix = False
    dataset_cpy.seq_duration = None

    bins = args.frame_length//2 + 1
    _min, _max = np.ones(bins)*float('inf'), np.ones(bins)*-float('inf')
    stft_obj = STFT(frame_length=args.frame_length, frame_step=args.frame_step)
    spec_obj = Spectrogram(mono=True)

    for index in tqdm(range(len(dataset_cpy)), desc='Compute dataset statistics'):

        x, y = dataset_cpy[index]

        # Add None to compensate batch_size
        features = spec_obj(stft_obj(x[None, ...]))
        _min = np.minimum(np.min(np.squeeze(features).T, axis=0), _min)
        _max = np.maximum(np.max(np.squeeze(features).T, axis=0), _max)

    return _min, _max

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--target', type=str, default='vocals',
                        help='target source')
    parser.add_argument('--verpose', type=bool, default=False, 
                        help='print optional description')

    # Dataset Parameters
    parser.add_argument('--root', type=str, required=True, 
                        help='root to dataset directory')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path folder name')
                        
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='number of batches')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--lr-decay-factor', type=float, default=0.3,
                        help='learning rate decay factor')
    parser.add_argument('--lr-decay-patience', type=float, default=50,
                        help='number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='L2 regularization')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--model', type=str,
                        help='path to a model checkpoint')
    parser.add_argument('--seq-duration', type=int, default=5,
                        help='duration of samples')
    parser.add_argument('--frame-length', type=int, default=4096,
                        help='length of the frame in samples')
    parser.add_argument('--frame-step', type=int, default=1024,
                        help='length of frame hop in samples')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in hertz(Hz)')
    parser.add_argument('--channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--output-norm', type=bool, default=False,
                        help='apply output normalization')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    args, _ = parser.parse_known_args()

    use_cuda = torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # To get static behaviour
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    train_dataset, valid_dataset = load_dataset(root=args.root, target=args.target, seq_duration=args.seq_duration)

    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    # scaler_mean, scaler_std = standard_scaler(args, train_dataset)
    scaler_mean, scaler_std = None, None
    unmix = UMX(
        frame_length=args.frame_length,
        frame_step=args.frame_step,
        rate=train_dataset.samplerate,
        n_channels=args.channels,
        n_hidden=args.hidden_size,
        input_mean=scaler_mean,
        input_scale=scaler_std,
        max_freq=args.bandwidth,
        fc_bias=True,
        output_norm=args.output_norm,
        n_layer=3
    ).to(device)

    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    # Add option to resume training from a checkpoint
    if args.model:

        with open(os.path.join(args.model, args.target+'_params.json'), 'r') as fin:
            res = json.load(fin)
        # Load Checkpoint
        checkpoint_dict = torch.load(os.path.join(args.model, args.target+'.chkpnt'), map_location=device)

        # Load Model and optimizer parameters
        unmix.load_state_dict(checkpoint_dict['model_state'])
        optimizer.load_state_dict(checkpoint_dict['optimitzer_state'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state'])
        
        # Train starting from the last epoch 
        pbar = tqdm.trange(checkpoint_dict['epoch'], checkpoint_dict['epoch']+args.epochs)
        train_losses = res['train_losses']
        valid_losses = res['valid_losses']
        best_loss = res['best_loss']
        best_epoch = res['best_epoch']
    else:
        pbar = tqdm.trange(1, args.epochs+1)
        train_losses = []
        valid_losses = []
        best_loss = None
        best_epoch = None
    
    for epoch in pbar:
        
        pbar.set_description('Training Epoch')

        # One pass
        train_loss = train(device, unmix, train_sampler, optimizer)
        valid_loss = valid(device, unmix, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # End of Epoch
        pbar.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        if best_loss == None:
            best_loss = valid_loss
        # Best epoch is whose loss is lower than best loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        checkpoint_dict = {
            'epoch': epoch+1,
            'best_loss': best_loss,
            'epoch_loss': valid_loss,
            'model_state': unmix.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }

        # Save current checkpoint
        save_checkpoint(
            checkpoint_dict=checkpoint_dict,
            best=valid_loss==best_loss,
            target=args.target,
            path=args.output
        )

        # Current results
        params = {
            'epochs_trained': epoch,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }

        with open(os.path.join(args.output, args.target+'_params.json'), 'w') as of:
            of.write(json.dumps(params))

    breakpoint()
