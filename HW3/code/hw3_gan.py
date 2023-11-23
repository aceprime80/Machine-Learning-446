import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw3_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=2, stride=1, padding=1, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2,out_channels=4, kernel_size=3, stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8,padding=0,kernel_size=3,stride=1)
        self.linear = nn.Linear(in_features=200,out_features=1)
        

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for i in self.children():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_uniform(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)
            if isinstance(i, nn.Linear):
                nn.init.kaiming_uniform(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)
        
        pass

    def forward(self, x):
        # TODO: complete forward function
        x1 = self.maxpool1(F.relu(self.conv1(x)))
        x2 = self.maxpool2(F.relu(self.conv2(x1)))
        x3 = F.relu(self.conv3(x2))
        # print(x3.shape)
        return self.linear(torch.reshape(x3,(x3.shape[0],x3.shape[1]*x3.shape[2]*x3.shape[3])))
        #return self.linear(torch.flatten(x3))


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.linear = nn.Linear(in_features=zdim,out_features=1568,bias=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1,bias=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1,bias=True)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=3,stride=1,padding=1,bias=True)
        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for i in self.children():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_uniform(i.weight.data)
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)
            if isinstance(i, nn.Linear):
                nn.init.kaiming_uniform(i.weight.data)
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0)
        
        
        pass

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        zlin = self.linear(z)
        #print(self.lrelu1(zlin).shape)
        z1 = self.conv1(self.up1(torch.reshape(self.lrelu1(zlin),(zlin.shape[0],32,7,7))))
        z2 = self.conv2(self.up2(self.lrelu2(z1)))
        z3 = F.sigmoid(self.conv3(self.lrelu3(z2)))

        return z3


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function
        # print("z size:")
        # print(z.shape)
        criterion = nn.BCEWithLogitsLoss()
        target = torch.cat((torch.zeros(batch_size, device=self._dev),torch.ones(batch_size, device=self._dev)))
        
        help = torch.ones((batch_size,1), device=self._dev)
        first = torch.flatten(self.disc(self.gen(z)))
        second = torch.flatten(self.disc(batch_data))
        mmm = torch.cat((first,second))
        return criterion(mmm,target)
        # for i in range(batch_size):
        #     loss -= criterion(self.disc(batch_data[i]),targetx)
        #     loss -= criterion(self.disc(self.gen(z[i])),targetz)
        #return loss
        

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        # TODO: implement generator's loss function
        # print("z size:")
        # print(z.shape)
        criterion = nn.BCEWithLogitsLoss()
        
        targetz = torch.ones((batch_size,1),device=self._dev)
        return criterion(self.disc(self.gen(z)), targetz)
        #for i in range(batch_size):
        #    loss -= criterion(self.disc(self.gen(z[i])), targetz)
        
        return loss
        

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
