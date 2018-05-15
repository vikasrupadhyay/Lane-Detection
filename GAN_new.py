import argparse
import os
import numpy as np
import math
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import transforms, utils
import cv2
import PIL

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
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
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

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
                   [transforms.Resize((32,32)),
                    # p.torch_transform(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


            image = self.transform(PIL.Image.fromarray(image))

        dictionary  ={}

        # print (image.shape)
        dictionary["image"] = np.array(image,dtype = float)
        dictionary["label"] = float(image_label)
        return dictionary


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data=torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data=torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        m.bias.data=torch.nn.init.constant(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks1 = nn.Sequential(nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv_blocks2 = nn.Sequential(nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8))
        self.conv_blocks3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh())
        # self.conv_blocks = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
        #     nn.Tanh()
        # )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks1(out)
        img = self.conv_blocks2(img)
        img = self.conv_blocks3(img)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(128*ds_size**2, opt.num_classes+1),
                                        nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
train_directory ='data_road/training/image_2'
test_directory = 'data_road/testing/image_2'
train_Data = KittiDataset(directory = train_directory, augment = False)
# os.makedirs('../../data/mnist', exist_ok=True)
dataloader = DataLoader(train_Data, batch_size=opt.batch_size, shuffle=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                         transforms.Resize(opt.img_size),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):

        imgs, labels = data['image'].view(len(data["label"]),3,32,32).float(),data['label'].float()
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss =  (adversarial_loss(real_pred, valid) + \
                        auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss =  (adversarial_loss(fake_pred, fake) + \
                        auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.data[0], 100 * d_acc,
                                                            g_loss.data[0]))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)