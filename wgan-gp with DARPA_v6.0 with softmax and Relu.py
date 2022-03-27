import argparse
import os
import numpy as np
import time
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

## 필독 https://study-grow.tistory.com/entry/Deep-learning-%EB%85%BC%EB%AC%B8-%EC%9D%BD%EA%B8%B0-StyleGAN-loss-%EC%9D%B4%ED%95%B4-%EC%96%95%EA%B2%8C-%EC%9D%BD%EB%8A%94-WGAN-WGAN-GP


min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

_time = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
os.makedirs("fake dataset", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=101, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=11, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=3, help="interval between image samples")
opt = parser.parse_args()
print(opt)

## Loss weight for gradient penalty ##
lambda_gp = 10

img_shape = (opt.channels, 1, opt.img_size)
img_size = opt.img_size
img_channel = opt.channels
batch_size = opt.batch_size

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



## Network traffic dataset ##
## smtp	4, web	2, ftp	3, ssh	1
def loading_dataset(): ## CSV Dataset load
    df = pd.read_csv('training dataset_week4.csv', index_col=0)
    dataset_df = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_ratio_trans_receive"]].astype(float)
    dataset_df = dataset_df.div(100)
    dataset_df[["ssh", "non-ssh"]] = df[["ssh", "non-ssh"]]
    tensor_data = torch.tensor(dataset_df.values)
    tensor_length = len(tensor_data)

    return (tensor_data, tensor_length)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.1, inplace=True)) # we change the gradient from 0.2 to 0.1 after 2021.12.16
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):  ## critic ##
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.1, inplace=True), # we change the gradient from 0.2 to 0.1 after 2021.12.16
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True), # we change the gradient from 0.2 to 0.1 after 2021.12.16
            nn.Linear(256, 1),
            ## WGAN-GP에서는 sigmoid 함수를 적용하지 않고 1-Lipschitz function 을 적용함
        )

    def forward(self, img):
        #print("img shape:", img.shape)
        img_flat = img.view(img.shape[0], -1)
        #print("img_flat shape:", img_flat.shape)
        validity = self.model(img_flat)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgan_gp(filename):

    ## dataset 생성 ##
    dataset, dataset_length = loading_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,)

    ## Initialize generator and discriminator ##
    generator = Generator()
    discriminator = Discriminator()

    # ## GPU Check ##
    # cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()

    ## Optimizers ##
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    #batches_done = 0



    header = pd.DataFrame(columns=["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_ratio_trans_receive", "ssh", "non-ssh", "epoch", "d_loss", "g_loss", "Destination", "Destination Port", "Destination_ip_port", "Gap of loss", "LABEL", "LABEL with GAP", "LABEL with GAP and Softmax"])
    with open(filename, "w") as f:
        header.to_csv(f, index=False)
        f.close()

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            ## dataset 차원 변환
            imgs = imgs.reshape((batch_size,img_channel,1,img_size))
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            #print("Real:", real_imgs.data.shape, "Fake:",real_imgs.data.shape)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                #if (epoch <=10 and epoch >= 5):
                    # 저장 전 fake_data 의 차원 변경 필요
                fake_imgs = fake_imgs.reshape((batch_size, img_size))
                fake_data = fake_imgs.detach().numpy() ## https://sanghyu.tistory.com/19 ##
                #fake_data = min_max_scaler.fit_transform(fake_data)
                fake_data = pd.DataFrame(fake_data)
                fake_data.columns = ["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_ratio_trans_receive", "ssh", "non-ssh"]
                fake_data["epoch"] = epoch # epoch, d_loss, g_loss, Destination, Destination_port, Destination_ip_port, gap of loss, LABEL, LABEL with GAP, LABEL with GAP and VAR, target var
                fake_data["d_loss"] = d_loss.item()
                fake_data["g_loss"] = g_loss.item()
                fake_data["Destination"] = "255.255.255.255"
                fake_data["Destination Port"] = "1"
                fake_data["Destination_ip_port"] = "255.255.255.255:1"
                fake_data["Gap of loss"] = fake_data["d_loss"] - fake_data["g_loss"]
                fake_data[["ssh", "non-ssh"]] = softmax(fake_data[["ssh", "non-ssh"]])
                fake_data["LABEL"] = "unknown"
                fake_data["LABEL with GAP"] = "unknown"
                fake_data["LABEL with GAP and Softmax"] = "unknown"

                with open(filename, "ab") as f:
                    f.write(b"\n")
                    fake_data.to_csv(f, index=False, header=None)

    df = pd.read_csv(filename)
    df.duplicated()
    df.to_csv(filename, index=True)


def softmax(output):
    output = np.exp(output)
    sum_exp_output = output.sum(axis = 1)
    # sum_exp_output = pd.Series(sum_exp_output)
    output["ssh"] = output["ssh"]/sum_exp_output
    output["non-ssh"] = output["non-ssh"] / sum_exp_output
    return output


def non_ssh_filtering(fake_data,filename_non_ssh, threshold, epoch_low, epoch_high):
    ## 수준별 SSH 라벨 정의 ##
    idx_gap_softmax = fake_data[((fake_data["non-ssh"]>threshold)  & (fake_data["epoch"] > epoch_low) & (fake_data["epoch"] < epoch_high))].index
    fake_data.at[idx_gap_softmax, "LABEL with GAP and Softmax"] = "non-ssh"
    fake_data["LABEL"] = fake_data["LABEL with GAP and Softmax"]


    ## SSH dataset 필터링 ##
    df_non_ssh = fake_data[fake_data["LABEL"] == "non-ssh"]
    df_non_ssh = df_non_ssh.reset_index()
    del df_non_ssh["index"]

    ## 0~100 scale 조정, 0이하 100초과 값 삭제 ##
    for cols in df_non_ssh.columns.tolist()[0:9]:
        df_non_ssh[cols] = df_non_ssh[cols] * 100
        idx_negative = df_non_ssh[(df_non_ssh[cols] < 0 | (df_non_ssh[cols] > 100))].index
        df_non_ssh = df_non_ssh.drop(idx_negative)
        df_non_ssh = df_non_ssh.reset_index()
        del df_non_ssh["index"]

    # df_non_ssh.to_csv(filename_non_ssh+"_softmax_%d_%s" %(threshold, epoch), index=False)

    return df_non_ssh


def ssh_filtering(fake_data, filename_ssh, threshold, epoch_low, epoch_high):

    ## 수준별 SSH 라벨 정의 ##
    # ssh = 0
    # idx_gap_softmax = fake_data[((fake_data["ssh"]>0.9) & (abs(fake_data["Gap of loss"])< 0.1) & (fake_data["epoch"]>10) & (fake_data["epoch"]<101))].index
    # ssh = 0
    idx_gap_softmax = fake_data[((fake_data["ssh"]>threshold) & (fake_data["epoch"]>epoch_low) & (fake_data["epoch"]<epoch_high))].index
    fake_data.at[idx_gap_softmax, "LABEL with GAP and Softmax"] = "ssh"
    fake_data["LABEL"] = fake_data["LABEL with GAP and Softmax"]

    ## SSH dataset 필터링 ##
    df_ssh = fake_data[fake_data["LABEL"] == "ssh"]
    df_ssh = df_ssh.reset_index()
    del df_ssh["index"]

    ## 0~100 scale 조정, 0이하 100초과 값 삭제 ##
    for cols in df_ssh.columns.tolist()[0:9]:
        df_ssh[cols] = df_ssh[cols]*100
        idx_negative = df_ssh[(df_ssh[cols]<0 | (df_ssh[cols]>100))].index
        df_ssh = df_ssh.drop(idx_negative)
        df_ssh = df_ssh.reset_index()
        del df_ssh["index"]

    # df_ssh.to_csv(filename_ssh+"_softmax_%d_%s" %(threshold, epoch), index=False)

    return df_ssh


if __name__ == '__main__':
    softmax_threshold = 0.75
    epoch = "40-100"
    epoch_low = 40
    epoch_high = 101
    # ## output file define ##
    filename = "epoch0~101_fake data of wgan-gp with softmax_5step_10gp_2021-12-16-22.csv"
    # filename = "epoch0~%d_fake data of wgan-gp with softmax_%dstep_%dgp_%s.csv" % (opt.n_epochs, opt.n_critic, lambda_gp, _time)
    filename_ssh = "ssh_epoch0~%d_fake data of wgan-gp with softmax_%dstep_%dgp_%s.csv" % (opt.n_epochs, opt.n_critic, lambda_gp, _time)
    filename_non_ssh = "non_ssh_epoch0~%d_fake data of wgan-gp with softmax_%dstep_%dgp_%s.csv" % (opt.n_epochs, opt.n_critic, lambda_gp, _time)

    #
    # ## original dataset for training ##
    df = pd.read_csv('training dataset_week4.csv', index_col=0)
    df_original = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "log_time_taken", "log_time_interval_mean", "log_time_interval_var", "log_ratio_trans_receive", "Destination", "Destination Port", "LABEL", "ssh", "non-ssh"]]
    #

    # fake ssh extraction ##
    # wgan_gp("1_"+filename)
    # wgan_gp("2_"+filename)
    # wgan_gp("3_"+filename)

    # with Pool(6) as p:
    #     p.map(wgan_gp,["1_"+filename,"2_"+filename,"3_"+filename,"4_"+filename,"5_"+filename,"6_"+filename])


    fake_dataset1 = pd.read_csv("1_"+filename, index_col=0)
    fake_dataset2 = pd.read_csv("2_"+filename, index_col=0)
    fake_dataset3 = pd.read_csv("3_"+filename, index_col=0)
    fake_dataset4 = pd.read_csv("4_"+filename, index_col=0)
    fake_dataset5 = pd.read_csv("5_"+filename, index_col=0)
    fake_dataset6 = pd.read_csv("6_"+filename, index_col=0)

    # for cols in fake_dataset1.columns.tolist()[17:21]:
    #     fake_dataset1[cols] = "unknown"
    #     fake_dataset2[cols] = "unknown"
    #     fake_dataset3[cols] = "unknown"
    #     fake_dataset4[cols] = "unknown"
    #     fake_dataset5[cols] = "unknown"
    #     fake_dataset6[cols] = "unknown"

    df_ssh1 = ssh_filtering(fake_dataset1, "1_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)
    df_ssh2 = ssh_filtering(fake_dataset2, "2_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)
    df_ssh3 = ssh_filtering(fake_dataset3, "3_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)
    df_ssh4 = ssh_filtering(fake_dataset4, "4_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)
    df_ssh5 = ssh_filtering(fake_dataset5, "5_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)
    df_ssh6 = ssh_filtering(fake_dataset6, "6_"+filename_ssh, softmax_threshold, epoch_low, epoch_high)

    df_non_ssh1 = non_ssh_filtering(fake_dataset1, "1_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)
    df_non_ssh2 = non_ssh_filtering(fake_dataset2, "2_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)
    df_non_ssh3 = non_ssh_filtering(fake_dataset3, "3_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)
    df_non_ssh4 = non_ssh_filtering(fake_dataset4, "4_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)
    df_non_ssh5 = non_ssh_filtering(fake_dataset5, "5_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)
    df_non_ssh6 = non_ssh_filtering(fake_dataset6, "6_" + filename_non_ssh, softmax_threshold, epoch_low, epoch_high)

    length_ssh = len(df_ssh1) + len(df_ssh2) + len(df_ssh3) + len(df_ssh4) + len(df_ssh5) + len(df_ssh6)
    length_non_ssh = len(df_non_ssh1) + len(df_non_ssh2) + len(df_non_ssh3) + len(df_non_ssh4) + len(df_non_ssh5) + len(df_non_ssh6)
    # length_non_ssh = len(df_non_ssh1)
    # length_ssh = len(df_ssh1)
    # length_non_ssh = len(df_non_ssh1)
    print("SSH:",length_ssh, " / NON-SSH:",length_non_ssh)

    # df_fake = pd.concat([df_ssh1,df_ssh2, df_ssh3, df_ssh4, df_ssh5, df_ssh6, df_non_ssh1,df_non_ssh2, df_non_ssh3, df_non_ssh4, df_non_ssh5, df_non_ssh6])
    df_fake = pd.DataFrame()
    df_fake_ssh = pd.concat([df_ssh1, df_ssh2, df_ssh3, df_ssh4, df_ssh5, df_ssh6])
    df_fake_non_ssh = pd.concat([df_non_ssh1, df_non_ssh2, df_non_ssh3, df_non_ssh4, df_non_ssh5, df_non_ssh6])
    # df_original = pd.concat([df_original,df_ssh1,df_ssh2, df_ssh3, df_ssh4, df_ssh5, df_ssh6, df_non_ssh1,df_non_ssh2, df_non_ssh3, df_non_ssh4, df_non_ssh5, df_non_ssh6]).reset_index()

    # df_fake_ssh.to_csv("fake dataset/"+"fake_ssh_%d_threshold_%f_epoch_%s.csv" % (length_ssh, softmax_threshold, epoch), index=False)
    df_fake_non_ssh.to_csv("fake dataset/"+"fake_non_ssh_%d_threshold_%f_epoch_%s.csv" % (length_non_ssh, softmax_threshold, epoch), index=False)

    # df_original.to_csv("fake dataset/training_dataset_with_fake_ssh_%d_non-ssh_%d with softmax.csv"  %(length_ssh, length_non_ssh), index = True)
    df_fake.to_csv("fake dataset/fake dataset_ssh_%d_non-ssh_%d with softmax_%f_%s.csv"  %(length_ssh, length_non_ssh, softmax_threshold, epoch), index = True)
    # df_original.to_csv("training_dataset_with_fake_non-ssh_%d.csv" % (length_non_ssh), index=True)

