from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torchvision.models as models
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
import PIL
import argparse
import random
import Augmentor
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Lane_detection_using_pretrained_resnet18')
parser.add_argument('--batch_size', type=int, default=4, metavar='B',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test_batch_size', type=int, default=4, metavar='TB',
                    help='input batch size for testing (default: 4)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.006, metavar='LR',
                    help='learning rate (default: 0.006)')

parser.add_argument('--valid_set_size', type=float, default=0.4, metavar='VSS',
                    help='validation set size (default: 0.4 so 4 % of all the batchs)')


parser.add_argument('--rot', type=int, default=0, metavar='RO',
                    help='1 for augmentation by rotation')
#setting Blurring for default augmentation
parser.add_argument('--gb', type=int, default=1, metavar='GB',
                    help='1 for augmentation by Gaussian Blurring')

parser.add_argument('--spk', type=int, default=0, metavar='SN',
                    help='1 for augmentation by Speckle Noise')

parser.add_argument('--isw', type=int, default=0, metavar='ISW',
                    help='1 for augmentation by Image Segmentation using watershed')

parser.add_argument('--shr', type=int, default=0, metavar='SH',
                    help='1 for augmentation by Shear')



parser.add_argument('--prob', type=float, default=0.5, metavar='PR',
                    help='Enter value between 0 and 1 for the probability by which augmentation is performed')


parser.add_argument('--noaug', type=int, default=1, metavar='NA',
                    help='Enter 0 for no Augmentation at all')



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(image),'label': torch.from_numpy(label)}

class KittiDataset(Dataset):
    def __init__(self, directory, augment = False, transform=True):
        directory = directory + "/*.png"
        self.img_names = list(glob.glob(directory))
        # print (self.img_names)
        self.transform = transform
        self.augment = augment
        # self.p = Augmentor.Pipeline(directory)
        # self.p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        # self.p.flip_left_right(probability=0.5)
        # self.p.flip_top_bottom(probability=0.5)


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self,idx):
        args = parser.parse_args()

        path = self.img_names[idx]
        image = cv2.imread(path)

        newHeight = 1200
        newWidth = 300
        # oldHeight = image.shape[0]
        # oldWidth = image.shape[1]
        # r = newHeight / oldHeight
        # newWidth = int(oldWidth * r)
        dim = (newHeight, newWidth)
        image = cv2.resize(image, dim,3, interpolation = cv2.INTER_AREA)
        # image = image.transpose(1,3)
        image_label = 0
        # print ("works")
        if 'uu_' in path:
            image_label = 0
        elif 'umm_' in path:
            image_label = 1
        elif 'um_' in path:
            image_label = 2
        else:
            print (" error in label")
            image_label = 2
        if self.augment:

            prob = args.prob

            if prob <0 or prob >1:
                prob =0.5

            #rotation of image 
            row,col,ch = 1200,300,3
            cv2.imwrite('image.png',image)
            if args.rot == 1 and np.random.uniform(0,1) > prob:
                angle = random.randint(1,80)
                M = cv2.getRotationMatrix2D((300/2,1200/2),angle,1)
                image = cv2.warpAffine(image.copy(),M,(300,1200))
            """*********************************************"""
            if args.gb == 1 and np.random.uniform(0,1) > prob:
            #Gaussian Blurring

                image = cv2.GaussianBlur(image,(5,5),0)


            """*********************************************"""
            #Segmentation algorithm using watershed
            if args.isw == 1 and np.random.uniform(0,1) > prob:

                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                # noise removal
                kernel = np.ones((3,3),np.uint8)
                opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=3)
                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)
                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)
                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0

                markers = cv2.watershed(image,markers)
                image[markers == -1] = [255,0,0]
                cv2.imwrite('Segmentation.png',image)

            """*********************************************"""

            #speckle noise

            if args.spk == 1 and np.random.uniform(0,1) > prob:
                row,col,ch = 1200,300,3
                gauss = np.random.randn(row,col,ch)
                gauss = gauss.reshape(row,col,ch)        
                image = image + image * gauss



            #HOG descriptor of a image

            # hog = cv2.HOGDescriptor()
            # image = hog.compute(image)

            #Shear transformation
            if args.shr == 1 :

                pts1 = np.float32([[5,5],[20,5],[5,20]])

                pt1 = 5+10*np.random.uniform()-10/2
                pt2 = 20+10*np.random.uniform()-10/2
                pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
                shear = cv2.getAffineTransform(pts1,pts2)

                image = cv2.warpAffine(image,shear,(col,row))
                cv2.imwrite('shear.png',image)


        if self.transform:
            self.transform = transforms.Compose(
                   [transforms.Resize((1200,300)),
                    # p.torch_transform(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])


            image = self.transform(PIL.Image.fromarray(image))

        dictionary  ={}

        # print (image.shape)
        dictionary["image"] = np.array(image,dtype = float)
        dictionary["label"] = float(image_label)
        return dictionary




def get_data_train():


    """ For creating the train labels have classified uu_ as 1, umm_ as 2 and um_ as 3"""
    train_images = []
    train_image_labels = []
    for img in glob.glob("data_road/training/image_2/*.png"):
        n= np.array(cv2.resize(cv2.imread(img),(224,224)))
        print (n.shape)
        train_images.append(n)
        if 'uu_' in img:
            train_image_labels.append(1)
        elif 'umm_' in img:
            train_image_labels.append(2)
        elif 'um_' in img:
            train_image_labels.append(3)
        else:
            print ("Noise in data")



    return np.array(train_image_labels).shape, np.array(train_images)




class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':
    # train_y,train_x = get_data_train()
    train_directory ='data_road/training/image_2'
    test_directory = 'data_road/testing/image_2'
    train_Data = KittiDataset(directory = train_directory, augment = False)
    correct_count = 0
    # for i in range(len(train_Data)):
    #   sample = train_Data[i]
    #   print (len(sample))
    train_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=0)


    

    input_dim = 1200 * 300
    batch_size = 32

    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    # mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    # dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                          shuffle=True, num_workers=2)

    print('Number of samples: ', len(train_dataloader))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(8, 100, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(2):
        for i, data in enumerate(train_dataloader, 0):
            inputs, classes = data['image'].view(len(data["label"]),3,1200,300).float(),data['label'].float()
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data[0]
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(1200, 300))
    plt.show(block=True)
