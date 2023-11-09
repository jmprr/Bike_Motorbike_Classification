# Bike_Motorbike_Classification
This project aims to classify input images as either bicycles or motorbikes using a deep convolutional neural network.

# Problem Statement
Given an image of a vehicle on the streets of Hanoi, classify whether it is a bicycle or motorbike. This is a binary image classification task.

# Data
The data consists of images scraped from online sources and manually labeled as either 'bike' or 'motorbike'. The images show bikes/motorbikes in traffic conditions on city streets.
The data is split into training, validation and test sets. Image augmentation techniques like random horizontal flipping are used to expand the training data.

# Model Architecture
A ResNet-18 model is used as the base network. The last fully connected layer is replaced with a 2 node output layer for the 2 classes.
The model is trained for 25 epochs using cross-entropy loss and SGD optimizer. Training includes calculating running loss, accuracy etc. The model is evaluated on the validation set after each epoch.

# Usage
* Install dependencies: 
pip install -r requirements.txt

* Data preprocessing: 
python3 dataset.py

* Train model: 
python3 train.py

* Inference: 
python3 test.py

# Performance
The model achieves 99,8% accuracy on the test set.
