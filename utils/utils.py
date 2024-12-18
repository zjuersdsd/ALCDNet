import random
import torchvision.datasets as datasets
from PIL import Image

import torch

from torch import nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import os
import random


def get_targets(filenames):
    targets = []

    for filename in filenames:
        text = filename.split('_', 3)
        text_ = text[2].split('.', 2)
        label = int(text_[0]) - 1


        targets.append(label)

    return targets

## 根据不同的通道之间的样本构建正负样本对
class EchoDataset(Dataset):
    def __init__(self, sound_dir="./dataset_tensor_106/", train=True, class_num=10, repeat=8, transform=None, target_transform=None, sequence_training=None, sequence_validation=None):
        self.transform = transform
        self.target_transform = target_transform
        self.class_num = class_num
        self.repeat = repeat
        self.train = train
        self.i_list = list(range(0, repeat))


        if self.train:
            self.sound_dir = sound_dir + "sequences/" + sequence_training + "/tensor/"
        else:
            self.sound_dir = sound_dir + "sequences/" + sequence_validation + "/tensor/"

        self.filenames = os.listdir(self.sound_dir)
        self.targets = get_targets(self.filenames)


    def __len__(self):
        return len(self.filenames)

    def get_loc(self, filename):
        text = filename.split('_', 3)
        text_ = text[2].split('.', 2)

        label = int(text_[0]) - 1
        ch =  int(text[0][3])

        return label, ch

    def __read_wav__(self, wav_path):
        data = torch.load(wav_path)
        return data
    #以dropout构建正负样本对
    # def __getitem__(self, idx):
    #     sound_path = os.path.join(self.sound_dir, self.filenames[idx])
    #     label,ch = self.get_loc(self.filenames[idx])
    #
    #
    #     sound_q = self.__read_wav__(sound_path)
    #     sound_p = sound_q
    #
    #     return sound_q,sound_p,label,ch



    #以相邻点构建正负样本对
    def __getitem__(self, idx):
        sound_path = os.path.join(self.sound_dir, self.filenames[idx])
        label,ch = self.get_loc(self.filenames[idx])
        if self.train == True:

            if (label+1) < self.class_num:
                pos_index = label + 2
            else:
                pos_index = label
            sound_q = self.__read_wav__(sound_path)
            sound_p = self.__read_wav__(os.path.join(self.sound_dir,
                                                     self.filenames[idx][0:4] + '_loc_' + str(pos_index) + '.pt'))
            # return sound_q,sound_p,label,
            # for moco
            return sound_q, sound_p, label, ch
            # ## for cnn
            # return sound_q,label
        else:

            sound_q = self.__read_wav__(sound_path)
            # for moco
            return sound_q,label,ch
            # ##for cnn
            # return sound_q, label

        # return pos_1,pos_2,label,ch
        # return pos_1,label
        #return pos_1,pos_2,label,i

    # 以不同通道的数据构建正负样本对
    # def __getitem__(self, idx):
    #     sound_path = os.path.join(self.sound_dir, self.filenames[idx])
    #     label,ch = self.get_loc(self.filenames[idx])
    #     ch_p
    #     if self.train == True:
    #
    #         if (label+1) < self.class_num:
    #             pos_index = label + 2
    #         else:
    #             pos_index = label
    #         sound_q = self.__read_wav__(sound_path)
    #         sound_p = self.__read_wav__(os.path.join(self.sound_dir,
    #                                                  self.filenames[idx][0:4] + '_loc_' + str(pos_index) + '.pt'))
    #         # return sound_q,sound_p,label,
    #         # for moco
    #         return sound_q, sound_p, label, ch
    #         # ## for cnn
    #         # return sound_q,label
    #     else:
    #
    #         sound_q = self.__read_wav__(sound_path)
    #         # for moco
    #         return sound_q,label,ch
    #         # ##for cnn
    #         # return sound_q, label
    #
    #     # return pos_1,pos_2,label,ch
    #     # return pos_1,label
    #     #return pos_1,pos_2,label,i
