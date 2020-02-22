import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class BlackBox(nn.Module):
    def __init__(self):
        super(BlackBox, self).__init__()
        self.predict_conv_net= nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.3),
            
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.3),           
                )
        self.predict_fc_net= nn.Sequential(                    
                    #Fully Connected Layers
                    # Output size after the two convolutional networks is (64,7,7)
                    nn.Linear(64*7*7, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256,10),
                )
        
    def forward(self, x):        
        x= x.view(-1, 1, 28, 28)
        out= self.predict_conv_net(x)
        out= out.view(-1,out.shape[1]*out.shape[2]*out.shape[3])
        return self.predict_fc_net(out)