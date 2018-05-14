
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




LATENT_DIM = 5 #size of the latent space in the variational autoencoder
BATCH_SIZE = 128

class VAE_simple(nn.Module):
    def __init__(self):
        super(VAE_simple, self).__init__()

        self.fc1 = nn.Linear(3*224*224, 400)
        self.extra_layer = nn.Linear(400, 100)
        self.extra_layer2 = nn.Linear(100, 100)
        self.fc21 = nn.Linear(100, LATENT_DIM)
        self.fc22 = nn.Linear(100, LATENT_DIM)
        
        self.fc3 = nn.Linear(LATENT_DIM, 100)
        self.extra_layer_dec = nn.Linear(100, 100)
        self.extra_layer_dec2 = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 3*224*224)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.extra_layer(h1))
        h1 = self.relu(self.extra_layer2(h1))
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
        h3 = self.relu(self.extra_layer_dec(h3))
        h3 = self.relu(self.extra_layer_dec2(h3))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3*224*224))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[4]:

def loss_function(reconstruced_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(reconstruced_x.view(-1, 3*224*224), x.view(-1, 3*224*224))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * (3*224*224)

    return BCE + KLD

def get_data_train():


	train_images = []
	train_image_labels = []
	names = []
	for img in glob.glob("data_road/training/image_2/*.png"):
		names.append(img)
		n= np.array(cv2.resize(cv2.imread(img),(224,224)))
		# print (n.shape)
		train_images.append(n)
		if 'uu_' in img:
			train_image_labels.append(1)
		elif 'umm_' in img:
			train_image_labels.append(2)
		elif 'um_' in img:
			train_image_labels.append(3)
		else:
			print ("Noise in data")



	return np.array(train_image_labels).shape, np.array(train_images),np.array(names)



def main():


	labels, image_ , names  = get_data_train()

	print (names)

	net = VAE_simple()
	net = torch.load('saved_model',map_location ='cpu')
	net.eval()

	for i in range(0,len(image_)):
		images = [0,0]
		imageact = image_[i]
		image = torch.from_numpy(imageact).view(3,224,224).float()
		if torch.cuda.is_available():
			output = net(Variable(image.unsqueeze(0).cuda()))
		else:
			output = net(Variable(image.unsqueeze(0)))
		images[0] = image #original image
		images[1] = output[0].data.view(3,224,224) # reconstructed image
	# cv2.imwrite('output1.png',images[1].cpu().numpy())
		torchvision.utils.save_image(images[1],'vaedataset/' + names[i]) 
		# torchvision.utils.save_image(images[0],'image1.png') 
		print ("done")

main()