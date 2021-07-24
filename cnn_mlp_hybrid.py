# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 2021

@author: mbadal1996
"""
# ==========================================================================
# CNN + MLP Classifier for Image + Numeric Data 
# ==========================================================================

# Comments:
# The following Python code is a hybrid CNN + MLP classifier for combined 
# image data + numeric features (meta-data) which further describe the images.
# The output of the model is a continuous float value in the range [0,1] which
# is due to normalization of the training label. In that sense it is a regression
# as opposed to a classification. The average error per epoch is calculated.
# The original purpose of the code was to make predictions on housing prices
# (see So-Cal Housing in Kaggle) but this kind of hybrid classifier is useful
# for various other problems where both images and numeric features are combined.
# In the event that a binary or multi-class output is desired (instead of a 
# float value regression), then the final output layer of the CNN+MLP should
# be modified for the number of classes and then passed through a softmax function.

# As an example, the house features (numeric data) CSV file is also included
# in the repository so that the user can see the format. House images are not
# included since they are too many and can be easily downloaded from Kaggle at:

# https://www.kaggle.com/ted8080/house-prices-and-images-socal

# Useful content at PyTorch forum is acknowledged for combining image 
# and numeric data features. 

# ----------------------------------------------------------

# IMPORTANT NOTE:
# When organizing data in folders to be input to dataloader, 
# it is important to keep in mind the following for correct loading:

# 1) The train and validation data were separated into their own folders by hand by 
# class (one class: house) called 'socal_pics/train' and 
# 'socal_pics/val'. That means the sub-folder 'train' 
# contains one folder: house. The same is true for the val data 
# held in the folder 'socal_pics/val'. So the organization looks like:

# socal_pics > train > house
# socal_pics > val > house

# Place the metadat CSV file in same folder as Python script

# 2) The test data is organized differently since there are no labels 
# for those images. Instead, the test data are held in the folder 
# 'socal_pics/test' where the sub-folder here 'test' 
# just contains one folder called 'test'. This is instead of the 'house' 
# folder. So the organization looks like:

# socal_pics > test > test

# =============================================================================

# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statistics as stat
#import copy

# Pytorch
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
#from torchvision.transforms import ToTensor
#from torch.utils.data import DataLoader

# ===========================================================================
# Parameters

# Image Parameters
CH = 3  # number of channels
ratio = 1.5625  # choose width/height ratio for resizing input images
imagewidth = 157  # size of image cropped to square (imagesize x imagesize) 
imageheight = int(np.floor(imagewidth/ratio))
cropsize = imageheight
#cropsize = imagewidth


# Neural Net Parameters
learn_rate = 5e-4  
num_epochs = 40  # About 70 epochs needed to converge for 100x100 images
batch_size = 100


# Seed for reproduceable random numbers (eg weights and biases)
# NOTE: Seed will be overidden by using image transforms like random flip 
# or setting shuffle = True in data loader.
torch.manual_seed(1234)


# ============================================================================

# Transforms
# Create transforms for training data augmentation. In each epoch, random 
# transforms will be applied according to the Compose function. They are random 
# since we are explicitly choosing "Random" versions of the transforms. To "increase
# the dataset" one should run more epochs, since each epoch has new random data.
# NOTE: Augmentation should only be applied to Training data.
# NOTE: When using augmentation transforms, it is best to use larger batches

# Transform for training data
transform_train = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees = (-20,20)), 
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()])

# Transform for validation data
transform_val = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor()])

# Transform for test data
transform_test = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor()])

# =========================================================================
# DATA IMPORT

# Import train,val, and test data and set up data loader.
# Note that ImageFolder will organize data according to class labels of the 
# folders "house, etc" as found in the train and val data folder.
# NOTE: When calling a specific image (such as 135) from train data, the first XXX
# images are class 0, then the next YYY are class 1, and etc. if more than one
# class existed (which is not the case here).

# Import CSV of Housing Data
# Read Data from File and Create Tensors for Train, Test, Validation
rawdata = pd.read_csv('socal2_cleaned_mod.csv')  # Train, val, test data

#  Import all columns in CSV
Xraw = np.column_stack((rawdata['image_id'].values,
                        rawdata['n_citi'].values,
                        rawdata['bed'].values,
                        rawdata['bath'].values,
                        rawdata['sqft'].values,
                        rawdata['price'].values))

# ==========================================================================
# Prepare Training Data

# ====================================================
# NOTE: Normalization was done after splitting data.
# ====================================================

Xraw_train = Xraw[0:2000,:]  # Pull out required amount of data for training
#Xraw_train = Xraw[0:650,:]  # Pull out required amount of data for training
#city_data_train = Xraw_train[:,1]  # import city of house to be used in training
bdrm_data_train = Xraw_train[:,2]  # import bdrm of house to be used in training
bath_data_train = Xraw_train[:,3]  # import bath of house to be used in training
sqft_data_train = Xraw_train[:,4]  # import sqft of house to be used in training
yraw_true_train = Xraw_train[:,5]  # import price of house for training/inference

# NORMALIZE DATA (COULD STANDARDIZE INSTEAD)
# Normalize data based to scale [0,1]. Could also standardize as z = (x - mean)/stddev
#city_train_norm = city_data_train/np.max(city_data_train)
bdrm_train_norm = bdrm_data_train/np.max(bdrm_data_train)
bath_train_norm = bath_data_train/np.max(bath_data_train)
sqft_train_norm = sqft_data_train/np.max(sqft_data_train)
y_true_train_norm = yraw_true_train/np.max(yraw_true_train)

# Convert to torch tensor
#city_train = torch.from_numpy(city_train_norm).float()
bdrm_train = torch.from_numpy(bdrm_train_norm).float()
bath_train = torch.from_numpy(bath_train_norm).float()
sqft_train = torch.from_numpy(sqft_train_norm).float()
y_train = torch.from_numpy(y_true_train_norm).float()

# Combine sqft, bdrm, etc into one meta_data
meta_train = torch.stack((bdrm_train,bath_train,sqft_train),dim=1)

# ===========================================================================
# Prepare Validation Data

# ===================================================
# NOTE: Normalization was done after splitting data. 
# ===================================================

Xraw_val = Xraw[2000:3000,:]  # Pull out required amount of data for val
#Xraw_val = Xraw[650:850,:]  # Pull out required amount of data for val
#city_data_val = Xraw_val[:,1]  # import city of house to be used in training
bdrm_data_val = Xraw_val[:,2]  # import bdrm of house to be used in val
bath_data_val = Xraw_val[:,3]  # import bath of house to be used in training
sqft_data_val = Xraw_val[:,4]  # import sqft of house to be used in val
yraw_true_val = Xraw_val[:,5]  # import price of house for inference

# NORMALIZE DATA (COULD STANDARDIZE INSTEAD)
# Normalize data based to scale [0,1]. Could also standardize as z = (x - mean)/stddev
#city_val_norm = city_data_val/np.max(city_data_val)
bdrm_val_norm = bdrm_data_val/np.max(bdrm_data_val)
bath_val_norm = bath_data_val/np.max(bath_data_val)
sqft_val_norm = sqft_data_val/np.max(sqft_data_val)
y_true_val_norm = yraw_true_val/np.max(yraw_true_val)

# Convert to torch tensor
#city_val = torch.from_numpy(city_val_norm).float()
bdrm_val = torch.from_numpy(bdrm_val_norm).float()
bath_val = torch.from_numpy(bath_val_norm).float()
sqft_val = torch.from_numpy(sqft_val_norm).float()
y_val = torch.from_numpy(y_true_val_norm).float()

# Combine sqft, bdrm, etc into one meta_data
meta_val = torch.stack((bdrm_val,bath_val,sqft_val),dim=1)


# ===========================================================================
# Generate batches of meta_data

# Metadata Training Batches
def get_batch_train(batch_size,which_batch,array_len=len(y_train)):
        
    num_batches = int(np.floor(array_len/batch_size))
    
    batch_y = []
    batch_meta = []
    for i in range(num_batches+1):
        batch_y_train = y_train[i*batch_size:(i+1)*batch_size]
        batch_y.append(batch_y_train)
        batch_meta_train = meta_train[i*batch_size:(i+1)*batch_size,:]
        # NOTE NOTE NOTE
        # NOTE: The line above should be enough to pull out batches directly
        # from meta_train tensor. Same with y_train. No need to store them in lists.
        batch_meta.append(batch_meta_train)
        
    ydata_train = torch.FloatTensor(batch_y[which_batch]) # call each batch
    metadata_train = torch.FloatTensor(batch_meta[which_batch])  # call each batch
    return ydata_train,metadata_train


# Metadata Validation Batches
def get_batch_val(batch_size,which_batch,array_len=len(y_val)):
        
    num_batches = int(np.floor(array_len/batch_size))
    
    batch_y = []
    batch_meta = []
    for i in range(num_batches+1):
        batch_y_val = y_val[i*batch_size:(i+1)*batch_size]
        batch_y.append(batch_y_val)
        batch_meta_val = meta_val[i*batch_size:(i+1)*batch_size,:]
        # NOTE NOTE NOTE
        # NOTE: The line above should be enough to pull out batches directly
        # from meta_val tensor. Same with y_val. No need to store them in lists.
        batch_meta.append(batch_meta_val)
        
    ydata_val = torch.FloatTensor(batch_y[which_batch]) # call each batch
    metadata_val = torch.FloatTensor(batch_meta[which_batch])  # call each batch
    return ydata_val,metadata_val


# ===========================================================================

# ===========================================================================
# Create Image Data Loader for Train, Validation, Test Images

# Training Data
# NOTE: Use "transform=transform if wanting data augmentation
images_train = datasets.ImageFolder('socal_pics/train',transform=transform_train)
loader_train = torch.utils.data.DataLoader(images_train, shuffle=False, batch_size=batch_size)

# Validation Data
images_val = datasets.ImageFolder('socal_pics/val',transform=transform_val)
loader_val = torch.utils.data.DataLoader(images_val, shuffle=False, batch_size=batch_size)

# Test Data
#images_test = datasets.ImageFolder('socal_pics/test',transform=transform_test)
#loader_test = torch.utils.data.DataLoader(images_test, shuffle=False, batch_size=len(images_val))


# ==========================================================================
# ==========================================================================

# Here we have used a combined CNN + MLP. The CNN processes image data and 
# the MLP is employed for input of numeric data. The outputs of each are 
# concatinated to form one stream of data.

# NOTE NOTE NOTE: The CNN used in this problem takes images of 100x100 pixels.
# if linear input layer is X * 22 * 22 or 200x200 pixels with X * 47 * 47.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # Image CNN
        self.conv1 = torch.nn.Conv2d(3, 10, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(10, 10, 5)
        self.fc1 = torch.nn.Linear(10 * 22 * 22, 120)
        self.fc2 = torch.nn.Linear(120, 60)
        #self.fc3 = torch.nn.Linear(60, 2)

        # Data MLP
        self.fc3 = torch.nn.Linear(3, 120)  # 3 inputs (eg bdrm,sqft,etc) to MLP
        self.fc4 = torch.nn.Linear(120, 60)

        # Cat outputs from CNN + MLP
        self.fc5 = torch.nn.Linear(60 + 60, 120)
        self.fc6 = torch.nn.Linear(120, 1)  # 1 ouput (price) from CNN+MLP

    def forward(self, x1, x2):
        # Image CNN
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, 10 * 22 * 22)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        
        # Data MLP
        x2 = x2.view(-1, 3)  
        x2 = F.relu(self.fc3(x2))
        x2 = F.relu(self.fc4(x2))
        
        # Cat outputs from CNN + MLP
        x3 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.fc5(x3))
        x3 = self.fc6(x3)
        
        return x3


# ===========================================================================
# ===========================================================================

# Call instance of CNN+MLP NN class
model = Net()

# MSE as our loss function since NN output is not a class but numeric in [0,1]
loss_fn = torch.nn.MSELoss(reduction='mean')

# Optimizer used to train parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)



# ===========================================================================

# Initialize tensor to store loss values
result_vals = torch.zeros(num_epochs,4)
count_train = 0  # Initialize Counter
count_val = 0
#X_test,y_true_test = loader_test

print(' ')
# print('epoch     ave_loss_train         ave_loss_val')
print('epoch     ave_loss_train         ave_loss_val          ave_error_train       ave_error_val')


error_train = torch.zeros(batch_size)
error_val = torch.zeros(batch_size)

for epoch in range(num_epochs):
    # New epoch begins
    running_loss_train = 0
    running_loss_val = 0
    running_error_train = 0
    running_error_val = 0
    num_batches_train = 0
    num_batches_val = 0
    count_train = 0
    count_val = 0
    j = 0  # Initialize batch counter
    k = 0  # Initialize batch counter
    
    model.train() # Set torch to train
    for X_train,_ in loader_train:
        # (X,y) is a mini-batch:
        # X size Nx3x32x32 (N: size mini-batch, 3: three colors, 32x32: height x width)
        # y size N
        
        # Get metadata in batches from function 
        ydata_train,metadata_train = get_batch_train(batch_size,j)
        
        # reset gradients to zero
        optimizer.zero_grad()
        
        # run model and compute loss
        N,C,nX,nY = X_train.size()
        y_pred_train = model(X_train.view(N,C,nX,nY),metadata_train)
        #y_pred_train = model(X_train.view(N,C,nX,nY))
        loss_train = loss_fn(y_pred_train.squeeze(), ydata_train)
        
        # back propagation
        loss_train.backward()
        
        # update the parameters
        optimizer.step()
        
        # compute  overall loss for entire training set
        running_loss_train += loss_train.detach().numpy()
        num_batches_train += 1 
       
        for i in range(len(ydata_train)):
            error_train[i] = abs(ydata_train[i].item() - y_pred_train.squeeze()[i].item())/ydata_train[i].item()
        error_train_sum = sum(error_train)
        running_error_train = running_error_train + error_train_sum
        
        j = j+1 # Step batch counter
        #print('j:',j)
        
    k = 0  # Re-initialize batch counter for validation
    model.eval()  # Set torch for evaluation
    for X_val,_ in loader_val:
        # (X,y) is a mini-batch:
        # X size Nx3x32x32 (N: size mini-batch, 3: three colors, 32x32: height x width)
        # y size N
        
        # Get metadata in batches from function
        ydata_val,metadata_val = get_batch_val(batch_size,k)
        
        # run model and compute loss
        N,C,nX,nY = X_val.size()
        y_pred_val = model(X_val.view(N,C,nX,nY),metadata_val)
        loss_val = loss_fn(y_pred_val.squeeze(), ydata_val)
        
        # compute overall loss for entire training set
        running_loss_val += loss_val.detach().numpy()
        num_batches_val += 1 
        
        for i in range(len(ydata_val)):
            error_val[i] = abs(ydata_val[i].item() - y_pred_val.squeeze()[i].item())/ydata_val[i].item()
        error_val_sum = sum(error_val)
        running_error_val = running_error_val + error_val_sum
        
        k = k+1  # Step batch counter
        #print('k:',k)

    ave_loss_train = running_loss_train/num_batches_train
    ave_loss_val = running_loss_val/num_batches_val
    ave_error_train = (running_error_train.item()/len(y_train))*100
    ave_error_val = (running_error_val.item()/len(y_val))*100


# ============================================

    
    # Store loss values to tensor "loss_vals" for later plotting
    result_vals[epoch, 0] = ave_loss_train  # loss per epoch
    result_vals[epoch, 1] = ave_loss_val  # loss per epoch
#    result_vals[epoch, 2] = ave_error_train  # accuracy per epoch
#    result_vals[epoch, 3] = ave_accuracy_val  # accuracy per epoch
    
    # Print loss every N epochs
    #if epoch % 2 == 1:
    print(epoch, '      ', ave_loss_train.item(), '  ', ave_loss_val.item(), '  ', ave_error_train, '  ', ave_error_val)
#    print(epoch, '      ', ave_loss_train.item(), '  ', ave_loss_val.item(), '  ', ave_error_train)
#    print(epoch, '      ', ave_loss_train.item(), '  ', ave_loss_val.item())

    
# ==========================================================================
# Plot Loss and Accuracy for train and val sets
# ==========================================================================

xvals = torch.linspace(0, num_epochs, num_epochs+1)
plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,0].detach().numpy())
plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,1].detach().numpy())
plt.legend(['loss_train', 'loss_val'], loc='upper right')
#plt.xticks(xvals[0:num_epochs])
plt.title('Loss (CNN + MLP Classifier)')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.tick_params(right=True, labelright=True)
plt.show()
#
#
# For plotting percent error (which needs to be added above)
#plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,2].numpy())
#plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,3].numpy())
#plt.legend(['error_train', 'error_val'], loc='lower right')
##plt.xticks(xvals[0:num_epochs])
#plt.title('Accuracy (CNN + MLP Classifier)')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.tick_params(right=True, labelright=True)
## plt.ylim(-0.15, 1.0)
#plt.show()
#
#
#
## ===========================================================================
