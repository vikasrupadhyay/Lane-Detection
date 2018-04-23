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




class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		return {'image': torch.from_numpy(image),'label': torch.from_numpy(label)}

class KittiDataset(Dataset):
	def __init__(self, directory, transform=None):
		directory = directory + "/*.png"
		self.img_names = list(glob.glob(directory))
		# print (self.img_names)
		self.transform = transform

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self,idx):

		path = self.img_names[idx]
		image = cv2.imread(path,0)

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


		if self.transform:

			image = self.transform(image)

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
		n= np.array(cv2.resize(cv2.imread(img),(1200,350)))
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





class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Sequential(         # input shape (3, 1200, 300)
			nn.Conv2d(
				in_channels=1,              
				out_channels=16,            
				kernel_size=5,              
				stride=1,                   
				padding=2,                  
			),                              
			nn.ReLU(),                      
			nn.MaxPool2d(kernel_size=2),    
		)
		self.conv2 = nn.Sequential(         
			nn.Conv2d(16, 32, 5, 1, 2),     
			nn.ReLU(),                      
			nn.MaxPool2d(2),                
		)
		self.out = nn.Linear(32 * 300 * 75, 100)   
		self.final = nn.Linear(100,3)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)           
		output = self.out(x)
		output = self.final(output)
		output = F.log_softmax(output, dim=1)
		return output    # return x for visualization if needed



def get_data_test():

	test_images = []
	for img in glob.glob("data_road/testing/image_2/*.png"):
		n= cv2.imread(img)
		test_images.append(n)



	test_caliberations =[]
	for calib in glob.glob("data_road/training/calib/*.txt"):

		n = pd.read_csv(calib, sep='\s{2,}', header=None)
		test_caliberations.append(n)

	print (len(train_images),len(train_label_images),len(test_caliberations))

def main():

	# train_y,train_x = get_data_train()
	train_directory ='data_road/training/image_2'
	test_directory = 'data_road/testing/image_2'
	train_Data = KittiDataset(directory = train_directory)
	correct_count = 0
	# for i in range(len(train_Data)):
	# 	sample = train_Data[i]
	# 	print (len(sample))
	train_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=4)


	model = Net().float()
	# model.train()

	optimizer = optim.Adam(model.parameters(),lr=0.06)   
	loss_func = nn.CrossEntropyLoss()                       

	for epoch in range(20):
		size = 0
		correct = 0
		for i_batch, sample in enumerate(train_dataloader):
			# print(i_batch, sample['label'].size(),sample['image'].size())
			image, label = Variable(sample["image"].view(len(sample["label"]),1,1200,300).float()), Variable(sample["label"].float())
			output = model(image)
			# print (output)
			# print (output.size(),label.size())
			optimizer.zero_grad()

			loss = loss_func(output, label.long())


			loss.backward()
			optimizer.step()


			pred_y = torch.max(output, 1)[1].data.squeeze()
			# print (pred_y,label)
			correct +=sum(np.array(pred_y)== np.array(label.data))
			size += float(label.size(0))
			# print ("**************************")
			# print (pred_y)
			# print (output)
			# print (correct)
			# print ("$$$$$$$$$$$$$$$$$$$$$$$$$$")

		accuracy = correct / size

		print ("Train accuracy for epoch ",epoch," is ", accuracy)


	test_Data = KittiDataset(directory = test_directory)


	test_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=4)

	accuracy = 0
	correct = 0
	size = 0
	for i_batch, sample in enumerate(test_dataloader):
			# print(i_batch, sample['label'].size(),sample['image'].size())
		image, label = Variable(sample["image"].view(len(sample["label"]),1,1200,300).float()), Variable(sample["label"].float())
		output = model(image)
	
		pred_y = torch.max(output, 1)[1].data.squeeze()
		correct +=sum(np.array(pred_y)== np.array(label.data))
		size += float(label.size(0))


	accuracy = correct / size

	print ("Test accuracy for model is ", accuracy)






main()