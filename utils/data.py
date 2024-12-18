import time

import torch
from torch.utils.data.dataloader import default_collate


from utils.utils import EchoDataset
import torch.utils.data


def datasets_concat_kitti(data_dir, train, data_transform, sequences_list):

    dataset_list = []

    for sequence in sequences_list:
        times_file = data_dir + "/sequences/" + sequence + "/wav_times.txt"

        time_stamps_len = 0
        with open(times_file, 'r') as f:

            for line in f.readlines():
                time_stamps_len = time_stamps_len + 1
        class_num = time_stamps_len
        d = EchoDataset(sound_dir=data_dir, train=train, class_num=class_num,
                                   transform=data_transform, sequence_training=sequence)

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list