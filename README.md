# cnn_mlp_hybrid_classifier
The following python / pytorch code uses a CNN+MLP Hybrid architecture to predict on image+numeric data.

=================================================================

CNN + MLP Classifier for Image + Numeric Data 


Comments:
The following Python code is a hybrid CNN + MLP classifier for combined 
image data + numeric features (meta-data) which further describe the images.
The output of the model is a continuous float value in the range [0,1] which
is due to normalization of the training label. In that sense it is a regression
as opposed to a classification. The original purpose of the code was to make 
predictions on housing prices (see So-Cal Housing in Kaggle) but this kind 
of hybrid classifier is useful for various other problems where both images
and numeric features are combined. In the event that a binary or multi-class
output is desired (instead of a float value regression), then the final 
output layer of the CNN+MLP should be modified for the number of classes 
and then passed through a softmax function.

As an example, the house features (numeric data) CSV file is also included
in the repository so that the user can see the format. House images are not
included since they are too many and can be easily downloaded from Kaggle at:

https://www.kaggle.com/ted8080/house-prices-and-images-socal

Useful content at PyTorch forum is acknowledged, which was helpful for 
combining image and numeric data features. 

----------------------------------------------------------

IMPORTANT NOTE:
When organizing data in folders to be input to dataloader, 
it is important to keep in mind the following for correct loading:

1) The train and validation data were separated into their own folders by hand by 
class (one class: house) called 'socal_pics/train' and 
'socal_pics/val'. That means the sub-folder 'train' 
contains one folder: house. The same is true for the val data 
held in the folder 'socal_pics/val'. So the organization looks like:

socal_pics > train > house

socal_pics > val > house

Place the metadata CSV file in same folder as Python script

2) The test data is organized differently since there are no labels 
for those images. Instead, the test data are held in the folder 
'socal_pics/test' where the sub-folder here 'test' 
just contains one folder called 'test'. This is instead of the 'house' 
folder. So the organization looks like:

socal_pics > test > test

============================================================
