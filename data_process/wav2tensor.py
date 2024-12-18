import os
import torchaudio
from tqdm import tqdm
import argparse
import random
import torch

from torch import nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import os



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_folder', default='./data/scene1', help='wave directory')
    parser.add_argument('--tensor_folder', default='./data/scene1', help='tensor directory')
    parser.add_argument('--n_fft', default=400, help='FFT points')
    parser.add_argument('--n_mels', default=64, help='n_mels')
    parser.add_argument('--win_length', default=None)
    parser.add_argument('--hop_length', default=100)
    args = parser.parse_args()

    wav_base_dir, tensor_base_dir, n_fft, win_length, hop_length = args.wav_folder, args.tensor_folder, args.n_fft, args.win_length, args.hop_length

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )


    for sequence in ["00","01","02"]:
        original_wav_dir = wav_base_dir +"/sequences/" + sequence +"/wav/"
        original_tensor_dir = tensor_base_dir + "/sequences/" + sequence + "/tensor/"
        filenames = os.listdir(original_wav_dir)
        with tqdm(total= len(filenames), mininterval=0.1, ncols=100) as pbar:
            for filename,i in zip(filenames,range(len(filenames))):

                wav_path = original_wav_dir + "/" + filename
                data, fs = torchaudio.load(wav_path)
                sound_q = spectrogram(data)
                filename_tensor = filename[0:-4] + ".pt"
                tensor_path = original_tensor_dir + "/" +filename_tensor
                torch.save(sound_q, tensor_path)
                pbar.update(1)