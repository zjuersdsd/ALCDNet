import argparse

import numpy as np
import pandas as pd
import torch
import yaml
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.backbone import ResNet18
from torchvision import transforms
from utils import utils
import torchvision.models as models
from evaluation.plot_PR_curve import compute_PR, compute_AP
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import faiss
np.set_printoptions(threshold=np.inf)
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
def generate_similarity_matrix(net, test_data_loader, loc_num, channel_num, feature_dim, audio_num=1):
    net.eval()
    target_list = torch.tensor([])
    feature_list = torch.tensor([])
    channel_list = torch.tensor([])
    similarity_matrix_final = torch.tensor([])
    index_one_list = []
    total_num = loc_num * audio_num
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
        feature_list, target_list, channel_list,similarity_matrix = feature_list.cuda(non_blocking=True), target_list.cuda(non_blocking=True), channel_list.cuda(
            non_blocking=True),similarity_matrix_final.cuda(non_blocking=True)
        for data, target, channel in test_bar:
            data, target, channel = data.cuda(non_blocking=True), target.cuda(non_blocking=True),channel.cuda(non_blocking=True)

            target_list = torch.cat((target_list, target), -1)
            channel_list = torch.cat((channel_list, channel), -1)
            feature, _ = net(data)
            feature_list = torch.cat((feature_list, feature), 0)

        _,channel_list_sorted_index = torch.sort(channel_list, dim=-1)
        target_list_sorted = torch.index_select(target_list,dim=0,index=channel_list_sorted_index)
        feature_list_sorted = torch.index_select(feature_list, dim=0, index=channel_list_sorted_index)

        target_list_sorted_reshaped = target_list_sorted.view(channel_num,total_num)
        feature_list_sorted_reshaped = feature_list_sorted.view(channel_num,total_num, feature_dim)

        for i in range(loc_num):
            i = i*audio_num
            index_one_list.append(i)

        for target_per_channel, feature_per_channel in zip(target_list_sorted_reshaped, feature_list_sorted_reshaped):
            _,target_per_channel_sorted_index= torch.sort(target_per_channel, dim=-1)
            feature_per_channel_sorted = torch.index_select(feature_per_channel, dim=0, index=target_per_channel_sorted_index)
            feature_per_channel_sorted_normed = feature_per_channel_sorted / torch.norm(feature_per_channel_sorted, dim=-1, keepdim=True)
            # similarity_matrix_per_channel = faiss.pairwise_distances(feature_per_channel_sorted_normed,feature_per_channel_sorted_normed, metric="METRIC_INNER_PRODUCT")

            similarity_matrix_per_channel = torch.mm(feature_per_channel_sorted_normed, feature_per_channel_sorted_normed.T)


            similarity_matrix_per_channel = torch.unsqueeze(similarity_matrix_per_channel, dim=0)
            similarity_matrix = torch.cat((similarity_matrix,similarity_matrix_per_channel), dim=0)

        return similarity_matrix

def get_poses(sound_dir, validation_sequence):

    poses_file = sound_dir + "sequences/" + validation_sequence + "/wav_poses.txt"

    poses2 = []
    with open(poses_file, 'r') as f:
        for x in f:
            x = x.strip().split()
            x = [float(v) for v in x]
            pose = torch.zeros((4, 4), dtype=torch.float64)
            pose[0, 0:4] = torch.tensor(x[0:4])
            pose[1, 0:4] = torch.tensor(x[4:8])
            pose[2, 0:4] = torch.tensor(x[8:12])
            pose[3, 3] = 1.0
            # pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
            poses2.append(pose.float().numpy())
    poses = poses2

    return poses


def main_process(args,distance_thr):

    sound_dir, checkpoints_dest, validation_sequence = args.sound_dir, args.checkpoints_dest, args.validation_sequence

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    cfg = cfg['experiment']
    feature_dim, m, temperature, momentum, gamma, alpha, channel_num = cfg['feature_dim'], cfg['m'], cfg['temperature'], \
    cfg['momentum'], cfg['gamma'], cfg['alpha'], cfg['channel_num']
    k, batch_size, epochs = cfg['k'], cfg['batch_size'], cfg['epochs']

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(32),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), ])

    times_file = sound_dir + "/sequences/" + validation_sequence + "/wav_times.txt"
    time_stamps_len = 0
    with open(times_file, 'r') as f:
        for line in f.readlines():
            time_stamps_len = time_stamps_len + 1
    class_num = time_stamps_len

    test_data = utils.EchoDataset(sound_dir=sound_dir, train=False, class_num=class_num,
                                  transform=data_transform, sequence_validation=validation_sequence)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    poses = np.stack(get_poses(sound_dir, validation_sequence))
    map_tree_poses = KDTree(np.stack(get_poses(sound_dir, validation_sequence))[:, :2, 3])


    model_q = ResNet18(feature_dim=feature_dim).cuda()
    if args.with_MoCo:
        if args.with_dtw:
            load_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size,
                                                                epochs,
                                                                gamma, alpha)
            load_path = checkpoints_dest + '{}_encoder_MoCo_sdtwloss_model.pth'.format(load_name_pre)
            state_dict = torch.load(load_path)
            model_q.load_state_dict(state_dict)
        else:
            load_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size,
                                                                epochs,
                                                                gamma, alpha)
            load_path = checkpoints_dest + '{}_encoder_MoCo_model.pth'.format(load_name_pre)
            state_dict = torch.load(load_path)
            model_q.load_state_dict(state_dict)
    else:
        load_name_pre = '{}_{}_{}'.format(feature_dim, batch_size, epochs)
        load_path = checkpoints_dest + '{}_encoder_model.pth'.format(load_name_pre)
        state_dict = torch.load(load_path)
        model_q.load_state_dict(state_dict)
        txt_path = checkpoints_dest + "pr_curve.txt"

    similarity_matrix = generate_similarity_matrix(model_q, test_loader, class_num, channel_num, feature_dim)
    similarity_matrix = torch.mean(similarity_matrix, dim=0, keepdim=False)
    print(similarity_matrix.shape)
    # similarity_matrix = (similarity_matrix[1]+similarity_matrix[3]+similarity_matrix[4]+similarity_matrix[5]+similarity_matrix[0]+similarity_matrix[6])/6
    _, _, precision_ours_fp, recall_ours_fp = compute_PR(similarity_matrix, poses, map_tree_poses,distance_thr,is_distance=False)

    ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
    print("Protocol 1 - Average Precision", ap_ours_fp)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ALCD')
    parser.add_argument('--sound_dir', default='./data', help='Tensor_ROOT')
    parser.add_argument('--checkpoints_dest', default='./checkpoints',
                        help='checkpoints_ROOT')
    parser.add_argument('--validation_sequence', type=str, default='02')
    parser.add_argument('--with_dtw', action='store_true', default=False,
                        help='Use soft-dtw loss')
    parser.add_argument('--with_MoCo', action='store_true', default=False,

                        help='Use MoCo')

    distance_thr = 0.3

    # args parse
    args = parser.parse_args()

    main_process(args,distance_thr)

    # main_process_orignal(args)


