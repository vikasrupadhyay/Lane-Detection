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


def new_feature_dataset_type():


	#use feature detection algorithm such as sobel and scharr, will be implementing !!!!!!
	return



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
                   [transforms.Resize((224,224)),
                   	# p.torch_transform(),
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


# currently validation is just the accuracy on the train set 
# Will create the splits later on
def train_model(model, criterion, optimizer, scheduler,dataloaders, num_epochs=10):

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	args = parser.parse_args()
	accuracies = []
	losses = []

	print ("Model running model on trian and validation set")
	number_of_batches = len(dataloaders)
	if args.valid_set_size > 8:
		args.valid_set_size = 8
	validation_set_size = number_of_batches - number_of_batches*args.valid_set_size*0.1
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('**********************************************************************')

		phase ='train'
		if phase == 'train':
			scheduler.step()
			model.train(True)  
		else:
			model.train(False)  

		curloss = 0.0
		correct = 0

		size = 0
		count = 0
		print (len(dataloaders))
		val_accuracies = []
		val_losses = []
		for data in tqdm(dataloaders):
			count +=1
			if count >= validation_set_size and phase != 'val':

				epoch_loss = curloss / size
				epoch_acc = correct / size

				val_accuracies.append(epoch_acc)
				val_losses.append(epoch_loss)
				print('{} Loss: {:.4f} Acc: {:.4f}'.format(
					phase, epoch_loss, epoch_acc))
				print (" Now running model on a validation set ")

				curloss = 0.0
				correct = 0

				size = 0

				phase ='val'

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
			# curloss += loss.item() * inputs.size(0)
			correct += torch.sum(preds == labels.data.long())
			size += len(labels)
		epoch_loss = curloss / size
		epoch_acc = correct / size
		accuracies.append(epoch_acc)
		losses.append(epoch_loss)

		print('{} Loss: {:.4f} Acc: {:.4f}'.format(
			phase, epoch_loss, epoch_acc))

		# deep copy the model
		if phase == 'val' and epoch_acc >= best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())

		print()

	print('Best val Acc: {:4f}'.format(best_acc))


	plt.plot(accuracies, list(range(1,10)))
	plt.plot(losses, list(range(1,10)))

	plt.plot(val_accuracies, list(range(1,10)))
	plt.plot(val_losses, list(range(1,10)))

	model.load_state_dict(best_model_wts)
	return model

def main():
	args = parser.parse_args()

	# train_y,train_x = get_data_train()
	train_directory ='data_road/training/image_2'
	test_directory = 'data_road/testing/image_2'
	train_Data = KittiDataset(directory = train_directory, augment = False)
	correct_count = 0
	# for i in range(len(train_Data)):
	# 	sample = train_Data[i]
	# 	print (len(sample))
	train_dataloader = DataLoader(train_Data, batch_size=4, shuffle=True, num_workers=4)


	model = models.resnet18(pretrained=True).float()
	num_epochs = args.epochs
	num_batchs_test = args.test_batch_size
	train_batch_size = args.batch_size
	learningrate = args.lr

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 3)

	if torch.cuda.is_available():
		model = model.cuda()

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.SGD(model.parameters(), learningrate, momentum=0.9)

	learningrate = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	model = train_model(model, criterion, optimizer, learningrate,
							train_dataloader,num_epochs)


	if args.noaug == 1:

		print ("Training with the following augmentations : ")

		if args.gb == 1:
			print ('* Gaussian Blurring')
		if args.rot == 1:
			print ('* Rotation')
		if args.shr == 1:
			print ('* Shear')

		if args.isw == 1:
			print ('* Image Segmentation with watershed')

		if args.spk == 1:

			print ("* Speckle Noise ")



		augmented_data = KittiDataset(directory = train_directory,augment = True)
		augmented_dataloader = DataLoader(augmented_data, batch_size=num_batchs_test, shuffle=True, num_workers=4)


		model = train_model(model, criterion, optimizer, learningrate,
								augmented_dataloader,num_epochs)


	test_Data = KittiDataset(directory = test_directory)


	test_dataloader = DataLoader(test_Data, batch_size=num_batchs_test, shuffle=True, num_workers=4)

	accuracy = 0
	correct = 0
	size = 0

	print ("**********************************")
	print ("Running model on Test Set")
	for i_batch, sample in tqdm(enumerate(test_dataloader)):
			# print(i_batch, sample['label'].size(),sample['image'].size())
		if torch.cuda.is_available():

			image, label = Variable(sample["image"].view(len(sample["label"]),3,224,224).float()).cuda(), Variable(sample["label"].float()).cuda()
		else:
			image, label = Variable(sample["image"].view(len(sample["label"]),3,224,224).float()), Variable(sample["label"].float())

		output = model(image)
	
		pred_y = torch.max(output, 1)[1].data.squeeze()
		correct +=sum(np.array(pred_y)== np.array(label.data))
		size += float(label.size(0))


	accuracy = correct / size

	print ("Test accuracy for model is ", accuracy)




if __name__ == '__main__':

    main()
