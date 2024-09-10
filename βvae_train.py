import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np

from model.βvae import BetaVAE_H
import lib.dataset as dset
from lib import utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



def run_train(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'dsprites':
        file = './datas/dsprites_dataset/dsprites.npz'
        transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])
        train_set = dset.Dspritesnpz(file, transform)
    elif args.dataset == 'CelebA':
        file = '../datas/CelebA'
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ])
        train_set = dset.CelebAFolder(file, transform)
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    model = BetaVAE_H(z_dim=args.z_dim, nc=args.input_dim, distribution=args.recon, mlp=True).to(device)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    # Initialize optimizer.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    var_list = np.array([])
    mu_var_list = np.array([])
    dimen_kl = np.array([])

    # Train model.
    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('vae_process:training')
        for i in range(args.epoch):
            model.train()
            var = 0
            mu = 0
            for j, images in enumerate(train_data):
                data = images.to(device)
                optimizer.zero_grad()

                total_dict = model(data)
                recon_vae = total_dict['recon_loss']

                kl = torch.tensor([0.]).to(device)
                dimension_kld = total_dict['dimension_wise_kld']
                kld = total_dict['kld_loss']

                var += total_dict['var'].mean(0).detach().cpu().numpy()
                mu += total_dict['mu'].var(0).detach().cpu().numpy()

                loss_vae = recon_vae + args.beta * kld
                if utils.isnan(loss_vae).any():
                    raise ValueError('NaN spotted in objective.')

                loss_vae.backward()
                optimizer.step()
            '''绘图'''
            var = var / len(train_data)
            mu = mu / len(train_data)
            dimen = dimension_kld.detach().cpu().numpy()
            var_list = np.append(var_list, var, 0)
            mu_var_list = np.append(mu_var_list, mu, 0)
            dimen_kl = np.append(dimen_kl, dimen, 0)

            print('\n kld={}, vae_recon={}'.format(kld.item(),
                                                   recon_vae.item()))

            print('\n all_var={}'.format(var))
            print('\n var_mu={}'.format(mu))
            pbar.update(1)
            if (i+1) % args.save_idx == 0:
                model.eval()
                save_path = './checkpoints'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(save_path, 'dsprites_vae{}.pt'.format(i+1)))
        var_list = var_list.reshape(args.epoch, -1)
        mu_var_list = mu_var_list.reshape(args.epoch, -1)
        dimen_kl = dimen_kl.reshape(args.epoch, -1)
        x = np.arange(0, args.epoch, 1)
        '''var'''
        # fig, ax = plt.subplots()
        # for v in range(10):
        #     y = var_list[:, v]
        #     ax.plot(x, y, label='z={}'.format(v))
        # ax.set_xlabel('epoch')  # 设置x轴名称 x label
        # ax.set_ylabel('var')  # 设置y轴名称 y label
        # ax.set_title('dimension_Var')  # 设置图名为Simple Plot
        # ax.legend()  # 自动检测要在图例中显示的元素，并且显示
        #
        # plt.show()  # 图形可视化
        # '''mu_var'''
        # fig, ax = plt.subplots()
        # for n in range(10):
        #     y = mu_var_list[:, n]
        #     ax.plot(x, y, label='z={}'.format(n))
        # ax.set_xlabel('epoch')  # 设置x轴名称 x label
        # ax.set_ylabel('mu_var')  # 设置y轴名称 y label
        # ax.set_title('dimension_mu_Var')  # 设置图名为Simple Plot
        # ax.legend()  # 自动检测要在图例中显示的元素，并且显示
        #
        # plt.show()  # 图形可视化
        # '''diemon'''
        fig, ax = plt.subplots()
        for n in range(10):
            y = dimen_kl[:, n]
            ax.plot(x, y, label='z={}'.format(n))
        ax.set_xlabel('Epoch')  # 设置x轴名称 x label
        ax.set_ylabel('element-wise KL')  # 设置y轴名称 y label
        ax.set_title('β-vae')  # 设置图名为Simple Plot
        ax.legend()  # 自动检测要在图例中显示的元素，并且显示
        plt.savefig('β-vae.pdf', format='pdf')
        plt.show()  # 图形可视化


