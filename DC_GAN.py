#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:53:01 2019

@author: matthew
"""

from __future__ import print_function
#%matplotlib inline
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# Set random seem for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# checking GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
def plot_losses(G_losses, D_losses):
    os.chdir("/home/matheus/IC/DC_GAN+CNN/aug/")
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training (C)")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("Generator and Discriminator Loss During Training (C)")
    
def animat(imgs):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imgs]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    HTML(ani.to_jshtml())
def comparison_RF(dataloader, img_list):
    os.chdir("/home/matheus/IC/DC_GAN+CNN/aug/")
    #Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.savefig("real_images_CELL-2")
    
    
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("fake_images_CELL-2")
    plt.show()
def save_G_images(bt_imgs, path ):
    os.chdir(path)
    cont = 1
    for i in bt_imgs:
       
        for j in range(i.shape[0]):
            save_image(i[j], "generated_"+str(cont)+".png", normalize=True)
            cont = cont + 1            
            
def save_models(G, D, path):
    torch.save(G.state_dict(), path + "GENEREATOR_MODEL2.pt")
    torch.save(D.state_dict(), path + "DISCRIMINATOR_MODEL2.pt")
        
def load_model(path, model):
    return model.load_state_dict(torch.load(path))


def main(argv):
    # Root directory for dataset
    dataroot = "/home/matheus/IC/datasets/cell-splited/train/c"
    
    
    # Batch size during training
    batch_size = 64
    
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    
    # Number of channels in the training images. For color images this is 3
    nc = 3
    
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    
    # Size of feature maps in generator
    ngf = 64
    
    # Size of feature maps in discriminator
    ndf = 64
    
    # Number of training epochs
    num_epochs = 80
    
    
    
   
    
    # transforms
    transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    # reading dataset
    dataset = datasets.ImageFolder(dataroot, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
    
    
    
    #========================================================================================
    #GENERATOR
    #========================================================================================
    
    # Create the generator
    netG = Generator(nz, ngf, nc).to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    
    #=========================================================================================
    #DISCRIMINATOR
    #=======================================================================================
    
    # create discriminator
    netD = Discriminator(nc, ndf).to(device)
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    #Loss function
    criterion = nn.BCELoss()
    
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # labels real or fake
    
    real_label = 1
    fake_label = 0
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.01, betas=(0.5, 0.999))
    
    img_list = [] # grid of images
    G_losses = [] # losses of generator
    D_losses = [] # losses discriminator
    iters = 0
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
    
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                
            
                
              
            
    
            iters += 1
     # ploting losses of generator and discriminartor      
    plot_losses(G_losses, D_losses)
    
    #animat(img_list)
    
    comparison_RF(dataloader, img_list)
   
    # saving models
    save_models(netG, netD, "/home/matheus/IC/datasets/aumentos/cell-info/")
    
    # loading model
    imgs = []
    for i in range(160):
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        with torch.no_grad():
            fake1 = netG(fixed_noise).detach().cpu()
            imgs.append(fake1)
    classes = ['Centromere', 'Golgi', 'Homogeneous', 'Nucleolar', 'NuMem', 'Speckled']
    #save_G_images(imgs, "/home/matheus/IC/datasets/aumentos/cell-splited/train/" + str(classes[2]) + "/")
    


    
    
if __name__ == "__main__":
    main(sys.argv[1:]) 