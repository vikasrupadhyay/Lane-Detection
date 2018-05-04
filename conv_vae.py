from __future__ import print_function
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
from torchvision.utils import save_image
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                    help='how big is z')
parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                    help='how big is linear around z')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



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
            if args.shr == 1 and np.random.uniform(0,1) > prob:

                pts1 = np.float32([[5,5],[20,5],[5,20]])

                pt1 = 5+10*np.random.uniform()-10/2
                pt2 = 20+10*np.random.uniform()-10/2
                pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
                shear = cv2.getAffineTransform(pts1,pts2)

                image = cv2.warpAffine(image,shear,(col,row))


        if self.transform:
            self.transform = transforms.Compose(
                   [#transforms.Resize((224,224)),
                    # p.torch_transform(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


            image = self.transform(PIL.Image.fromarray(image))

        dictionary  ={}

        # print (image.shape)
        dictionary["image"] = np.array(image,dtype = float)
        dictionary["label"] = float(image_label)
        return dictionary



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 300, 1200)
    return x

train_directory ='data_road/training/image_2'
test_directory = 'data_road/testing/image_2'
train_Data = KittiDataset(directory = train_directory, augment = False)
correct_count = 0
# for i in range(len(train_Data)):
#   sample = train_Data[i]
#   print (len(sample))
train_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=4)
test_Data = KittiDataset(directory = test_directory)
test_dataloader = DataLoader(test_Data, batch_size=4, shuffle=True, num_workers=4)



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1200 * 300 * 32, args.intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(args.intermediate_size, args.hidden_size)
        self.fc22 = nn.Linear(args.intermediate_size, args.hidden_size)

        # Decoder
        self.fc3 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc4 = nn.Linear(args.intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        print (mu.size(),logvar.size())
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, sample in enumerate(train_dataloader):
        if args.cuda:
            image, label = Variable(sample["image"].view(len(sample["label"]),3,1200,300).float()).cuda(), Variable(sample["label"].float()).cuda()
        else:
            image, label = Variable(sample["image"].view(len(sample["label"]),3,1200,300).float()), Variable(sample["label"].float())

        data = image
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_dataloader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if epoch == args.epochs and i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                   recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       'snapshots/conv_vae/reconstruction_' + str(epoch) +
                       '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch == args.epochs:
        sample = Variable(torch.randn(64, args.hidden_size))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 3, 32, 32),
                   'snapshots/conv_vae/sample_' + str(epoch) + '.png')
