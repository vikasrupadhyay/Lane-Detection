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






class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		return {'image': torch.from_numpy(image),'label': torch.from_numpy(label)}

class KittiDataset(Dataset):
	def __init__(self, directory, transform=True):
		directory = directory + "/*.png"
		self.img_names = list(glob.glob(directory))
		# print (self.img_names)
		self.transform = transform

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self,idx):

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


		if self.transform:
			self.transform = transforms.Compose(
                   [transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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





# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Sequential(         # input shape (3, 1200, 300)
# 			nn.Conv2d(
# 				in_channels=1,              
# 				out_channels=16,            
# 				kernel_size=5,              
# 				stride=1,                   
# 				padding=2,                  
# 			),                              
# 			nn.ReLU(),                      
# 			nn.MaxPool2d(kernel_size=2),    
# 		)
# 		self.conv2 = nn.Sequential(         
# 			nn.Conv2d(16, 32, 5, 1, 2),     
# 			nn.ReLU(),                      
# 			nn.MaxPool2d(2),                
# 		)
# 		self.out = nn.Linear(32 * 300 * 75, 100)   
# 		self.final = nn.Linear(100,3)

# 	def forward(self, x):
# 		x = self.conv1(x)
# 		x = self.conv2(x)
# 		x = x.view(x.size(0), -1)           
# 		output = self.out(x)
# 		output = self.final(output)
# 		output = F.log_softmax(output, dim=1)
# 		return output    # return x for visualization if needed



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


# currently validation is just the accuracy on the train set 
# Will create the splits later on
def train_model(model, criterion, optimizer, scheduler,dataloaders, num_epochs=25):

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train(True)  
			else:
				model.train(False)  

			curloss = 0.0
			correct = 0

			size = 0
			for data in dataloaders:
				inputs, labels = data['image'].view(len(data["label"]),3,224,224).float(),data['label'].float()

				if torch.cuda.is_available():
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				optimizer.zero_grad()

				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels.long())

				if phase == 'train':
					loss.backward()
					optimizer.step()

                # statistics
				curloss += loss.data[0] * inputs.size(0)
				correct += torch.sum(preds == labels.data.long())
				size += len(labels)
			epoch_loss = curloss / size
			epoch_acc = correct / size

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model

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


	model = models.resnet18(pretrained=True).float()

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 3)

	if torch.cuda.is_available():
		model = model.cuda()

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	learningrate = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	model = train_model(model, criterion, optimizer, learningrate,
							train_dataloader,num_epochs=25)


	test_Data = KittiDataset(directory = test_directory)


	test_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=4)

	accuracy = 0
	correct = 0
	size = 0
	for i_batch, sample in enumerate(test_dataloader):
			# print(i_batch, sample['label'].size(),sample['image'].size())
		if torch.cuda.is_available():

			image, label = Variable(sample["image"].view(len(sample["label"]),3,1200,300).float()).cuda(), Variable(sample["label"].float()).cuda()
		else:
			image, label = Variable(sample["image"].view(len(sample["label"]),3,1200,300).float()), Variable(sample["label"].float())

		output = model(image)
	
		pred_y = torch.max(output, 1)[1].data.squeeze()
		correct +=sum(np.array(pred_y)== np.array(label.data))
		size += float(label.size(0))


	accuracy = correct / size

	print ("Test accuracy for model is ", accuracy)





main()