#pytorch imports
import torch
import torch.nn as nn

#CNN
class CNN(nn.Module): 
    def __init__(self): #constructor for our CNN
        super(CNN, self).__init__()

        #Convolutional Layers (maps pixel value using kernals to map what the image looks like,taking numerical values using the kernel as a key to represent the image)
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) #basic local features- edges, color contrast, corners etc
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1) #simple patterns- shapes created by textures in layer one, textures
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1) #high level patterns- objects, more complex shapes/textures, etc

        #ppooling layer
        self.pool = nn.MaxPool2d(2, 2) #condenses info into an integer to simplify and reduce memory usage

        #fully connected layer
        self.fc = nn.Linear(64 * 4 * 4, 10) #connects the pooled information and creates an image

    def forward(self, x):
        #applies convolution layers and pools information
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        #converts each image into a vector
        x = x.view(-1, 64 * 4 * 4)

        #converts vector into a 10 ouput vector (measures how likely the image is to belong to each class)
        x = self.fc(x)

        return x