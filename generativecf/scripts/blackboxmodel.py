import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


class BlackBox(nn.Module):
		def __init__(self, inp_shape):
		    
		    super(BlackBox, self).__init__()
		    self.inp_shape = inp_shape
		    self.hidden_dim = 10		    
		    self.predict_net= nn.Sequential(
		                 nn.Linear( self.inp_shape, self.hidden_dim),
		                 nn.Linear( self.hidden_dim, 2),
		                )
		def forward(self, x):
		    return self.predict_net(x)