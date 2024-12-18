import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils import utils
from model.backbone import ResNet18

import torchvision.models as models

from utils.soft_dtw_cuda import SoftDTW
from utils.data import datasets_concat_kitti
import yaml
import random
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

seed = 150
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# train for one epoch to learn unique features
def train_with_dtw(encoder_q, encoder_k, data_loader, train_optimizer, sdtw, alpha, temperature, momentum, epoch, epochs):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _, _ in train_bar:

        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        query,h_q = encoder_q(x_q)
        h_q = h_q.unsqueeze(dim=2)
        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)

        key,h_k = encoder_k(x_k[idx])
        h_k = h_k.unsqueeze(dim=2)


        key = key[torch.argsort(idx)]
        loss_sdtw = sdtw(h_q,h_k)
        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)


        score_neg = torch.mm(query, memory_queue.t().contiguous())

        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        loss = loss + alpha * loss_sdtw.mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def train_without_dtw(encoder_q, encoder_k, data_loader, train_optimizer, temperature, momentum, epoch, epochs):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _, _ in train_bar:

        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        query,h_q = encoder_q(x_q)
        h_q = h_q.unsqueeze(dim=2)
        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        key,h_k = encoder_k(x_k[idx])
        h_k = h_k.unsqueeze(dim=2)


        key = key[torch.argsort(idx)]
        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num



def train_without_MoCo(encoder_q, data_loader, train_optimizer):

    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _, _ in train_bar:

        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        query,_ = encoder_q(x_q)
        key, _ = encoder_q(x_k)

        out = torch.cat([query,key],dim=0)

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(query * key, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ALCD')
    parser.add_argument('--sound_dir', default='./data/2024_4_23_underground_395/', help='Tensor_ROOT')
    parser.add_argument('--checkpoints_dest', default='./checkpoints/2024_4_23_underground_395/', help='checkpoints_ROOT')
    parser.add_argument('--with_dtw', action='store_true', default=False,
                        help='Use soft-dtw loss')
    parser.add_argument('--with_MoCo', action='store_true', default=False,
                        help='Use MoCo')
    # args parse
    args = parser.parse_args()

    sound_dir, checkpoints_dest = args.sound_dir, args.checkpoints_dest

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    cfg = cfg['experiment']
    feature_dim, m, temperature, momentum, gamma, alpha = cfg['feature_dim'], cfg['m'], cfg['temperature'], cfg[
        'momentum'], cfg['gamma'], cfg['alpha']
    k, batch_size, epochs = cfg['k'], cfg['batch_size'], cfg['epochs']

    sequences_training = ["01","02"]

    print("Using sequences", sequences_training, "to train")

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(32),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), ])

    train_data,_ = datasets_concat_kitti(sound_dir,True,data_transform,sequences_training)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)

    # model setup and optimizer config
    # model_q = Model(feature_dim=256,base_encoder=models.__dict__["resnet18"],mlp = True).cuda()
    # model_k = Model(feature_dim=256,base_encoder=models.__dict__["resnet18"],mlp = True).cuda()

    # model setup 2

    model_q = ResNet18(feature_dim=feature_dim).cuda()
    model_k = ResNet18(feature_dim=feature_dim).cuda()

    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    # init memory queue as unit random vector ---> [M, D]
    # soft_dtw_loss
    if args.with_MoCo:
        print("training with MoCo")
        if args.with_dtw:
            print("training with soft-dtw loss")
            sdtw = SoftDTW(use_cuda=True, gamma=gamma, normalize=True)

            memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)
            mini_loss = 10000
            save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs,
                                                            gamma, alpha)
            train_loss_list = []
            for epoch in range(1, epochs + 1):
                train_loss = train_with_dtw(model_q, model_k, train_loader, optimizer, sdtw, alpha, temperature, momentum, epoch, epochs)
                train_loss_list.append(train_loss)

                if train_loss < mini_loss:
                    mini_loss = train_loss
                    print("save model")
                    save_path = checkpoints_dest + '{}_encoder_MoCo_sdtwloss_model.pth'.format(save_name_pre)
                    torch.save(model_q.state_dict(), save_path)
            loss_path = checkpoints_dest + "{}_encoder_MoCo_sdtwloss_train_loss.txt".format(save_name_pre)
            with open(loss_path, 'w') as train_loss:
                train_loss.write(str(train_loss_list))

        else:
            print("training without soft-dtw loss")

            memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)
            mini_loss = 10000
            save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size,
                                                                epochs,
                                                                gamma, alpha)
            train_loss_list = []
            for epoch in range(1, epochs + 1):
                train_loss = train_without_dtw(model_q, model_k, train_loader, optimizer, temperature, momentum, epoch,
                                   epochs)
                train_loss_list.append(train_loss)

                if train_loss < mini_loss:
                    mini_loss = train_loss
                    print("save model")
                    save_path = checkpoints_dest + '{}_encoder_MoCo_model.pth'.format(save_name_pre)
                    torch.save(model_q.state_dict(), save_path)
            loss_path = checkpoints_dest + "{}_encoder_MoCo_train_loss.txt".format(save_name_pre)
            with open(loss_path, 'w') as train_loss:
                train_loss.write(str(train_loss_list))
    else:
        print("training without MoCo")
        mini_loss = 10000
        save_name_pre = '{}_{}_{}'.format(feature_dim, batch_size, epochs)
        train_loss_list = []
        for epoch in range(1, epochs + 1):
            train_loss = train_without_MoCo(model_q, train_loader, optimizer)
            train_loss_list.append(train_loss)

            if train_loss < mini_loss:
                mini_loss = train_loss
                print("save model")
                save_path = checkpoints_dest + '{}_encoder_model.pth'.format(save_name_pre)
                torch.save(model_q.state_dict(), save_path)
        loss_path = checkpoints_dest + "{}_encoder_train_loss.txt".format(save_name_pre)
        with open(loss_path, 'w') as train_loss:
            train_loss.write(str(train_loss_list))


