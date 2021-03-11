import argparse
import os
import numpy as np
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
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()



os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=10, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=3, help="interval between image samples")
opt = parser.parse_args()
print(opt)


img_shape = (opt.channels, 1, opt.img_size)
img_size = opt.img_size
img_channel = opt.channels
batch_size = opt.batch_size

## Network traffic dataset ##
## smtp	4, web	2, ftp	3, ssh	1
def loading_dataset(): ## CSV Dataset load
    df = pd.read_csv('df_training_20210310_1normalization.csv', index_col=0)
    dataset_df = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time",
                     "log_time_taken", "log_ratio_trans_receive", "no_url", "target"]]
    #X = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    #dataset_df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time","log_time_taken", "no_url", "log_ratio_trans_receive","target"]] = min_max_scaler.fit_transform(dataset_df.values)
    #dataset_df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time", "log_time_taken", "no_url", "log_ratio_trans_receive","target"]] = standard_scaler.fit_transform(dataset_df.values)

    tensor_data = torch.tensor(dataset_df.values)
    #tensor_data = standard_scaler.fit_transform(tensor_data)
    #tensor_target = torch.tensor(dataset_df["target"])
    tensor_length = len(tensor_data)
    return (tensor_data, tensor_length)


dataset, dataset_length = loading_dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,)
######################################################################

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        #print("img shape:", img.shape)
        img_flat = img.view(img.shape[0], -1)
        #print("img_flat shape:", img_flat.shape)
        validity = self.model(img_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()



# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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


# ----------
#  Training
# ----------

batches_done = 0
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

            if ((epoch+1)%25 == 0):
                # 저장 전 fake_data 의 차원 변경 필요
                fake_imgs = fake_imgs.reshape((batch_size, img_size))
                fake_data = fake_imgs.detach().numpy() ## https://sanghyu.tistory.com/19 ##
                fake_data = min_max_scaler.fit_transform(fake_data)
                fake_data = pd.DataFrame(fake_data)
                fake_data.columns = ["log_time_taken", "log_cs_byte", "log_ratio_trans_receive", "log_count_connect_IP", "log_count_total_connect", "log_avg_count_connect", "log_transmit_speed_BPS",
                                     "Business.time", "no_url", "target"]
                fake_data["epoch"] = epoch
                fake_data["d_loss"] = d_loss.item()
                fake_data["g_loss"] = g_loss.item()
                fake_data["Destination"] = "255.255.255.255"
                fake_data["Destination Port"] = "1"
                fake_data["Destination_ip_port"] = "255.255.255.255:1"


                                ## d_loss 가 크고, g_loss가 작은 경우가 유용한 샘플로 사용 가능
                with open("fake_data.csv", "ab") as f:
                    f.write(b"\n")
                    #f.write(b"[Epoch %d/%d][Batch %d/%d],[D loss: %f],[G loss: %f],[D loss - G loss: %f]\n" % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), d_loss-g_loss))
                    fake_data.to_csv(f, index=False)

            ## i) 최적 모델링을 위한 WEIGHT 저장
            ## ii) SSH에 대한 충분한 샘플링은 어떻게?
            ## iii)

            batches_done += opt.n_critic
