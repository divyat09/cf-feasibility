import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class CF_VAE(nn.Module):
    
    def __init__(self, data_size, encoded_size):

        super(CF_VAE, self).__init__()
        
        self.data_size = data_size
        self.encoded_size= encoded_size
        # size= (W-Kernel_Size)/Stride + 1 after one conv2d; downsampling by factor of 2 in case of maxpool
        self.convoluted_size=int( (self.data_size - 3*2) ) 

        # No Categorical Variables for MINST, everything is continuous 
        self.encoded_categorical_feature_indexes= []     
        self.encoded_continuous_feature_indexes=[]
        self.encoded_start_cat= 0
        
        self.encoder_mean_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            )

#         self.encoder_mean_conv= nn.Sequential(
#             nn.Linear( 1*28*28, 2*self.convoluted_size**2),
#             nn.BatchNorm1d(2*self.convoluted_size**2),
#             nn.ReLU(),
#             nn.Linear( 2*self.convoluted_size**2, self.convoluted_size**2),
#             nn.BatchNorm1d(self.convoluted_size**2),
#             nn.ReLU(),
#             )

   
        self.encoder_mean_fc = nn.Sequential(
        # Plus 1 to the input encoding size and data size to incorporate the target class label        
            nn.Linear( self.convoluted_size**2+1, self.encoded_size),
            nn.BatchNorm1d(self.encoded_size),            
            #nn.ReLU(),
            #nn.Linear(self.encoded_size, self.encoded_size),
            #nn.BatchNorm1d(self.encoded_size),
            #nn.ReLU(),
            )
        
        self.encoder_var_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            )

#         self.encoder_var_conv= nn.Sequential(
#             nn.Linear( 1*28*28, 2*self.convoluted_size**2 ),
#             nn.BatchNorm1d(2*self.convoluted_size**2),
#             nn.ReLU(),
#             nn.Linear( 2*self.convoluted_size**2, self.convoluted_size**2),
#             nn.BatchNorm1d(self.convoluted_size**2),
#             nn.ReLU(),
#         )
 
        self.encoder_var_fc = nn.Sequential(
            # Plus 1 to the input encoding size and data size to incorporate the target class label                    
            nn.Linear( self.convoluted_size**2+1, self.encoded_size),
            nn.BatchNorm1d(self.encoded_size),
            #nn.ReLU(),
            #nn.Linear(self.encoded_size, self.encoded_size),
            #nn.BatchNorm1d(self.encoded_size),
            nn.Sigmoid(),
            )
 
        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean_fc = nn.Sequential(
            nn.Linear( self.encoded_size+1, self.convoluted_size**2 ),
            nn.BatchNorm1d(self.convoluted_size**2),
            nn.ReLU(),
            #nn.Linear( self.convoluted_size**2, self.convoluted_size**2 ),
            #nn.BatchNorm1d(self.convoluted_size**2),
            #nn.ReLU(),
            )

        self.decoder_mean_conv= nn.Sequential(
            nn.Linear( self.convoluted_size**2, 2*self.convoluted_size**2),
            nn.BatchNorm1d(2*self.convoluted_size**2),
            nn.ReLU(),
            nn.Linear( 2*self.convoluted_size**2, 1*28*28 ),
            nn.BatchNorm1d(1*28*28),
            nn.Sigmoid(),
        ) 
        
#         self.decoder_mean_conv = nn.Sequential(
#             nn.ConvTranspose2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
#             nn.ReLU(),
#             #nn.MaxUnpool2d(kernel_size=2, stride=2),
#             nn.ConvTranspose2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
#             nn.Sigmoid(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0),
#             nn.Sigmoid(),
#             )

        
    def encoder(self, x, c):
        x= x.view(-1, 1, self.data_size, self.data_size)
        
        mean= self.encoder_mean_conv(x)
        mean= mean.view(-1, self.convoluted_size**2)
        mean= torch.cat((mean, c), 1)
        mean= self.encoder_mean_fc(mean)
        
        logvar= self.encoder_var_conv(x)
        logvar= logvar.view( -1, self.convoluted_size**2) 
        logvar= torch.cat((logvar, c),1)
        logvar = 0.01+ self.encoder_var_fc(logvar)
        
        return mean, logvar

    def decoder(self, z, c):
        z=torch.cat((z, c),1)
        mean= self.decoder_mean_fc(z)
        #mean= mean.view(-1, 1, self.convoluted_size, self.convoluted_size)
        mean= self.decoder_mean_conv(mean)
        
        mean= mean.view(-1, self.data_size**2)
        
        return mean
    
    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum( -.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar) ), axis=1)
                          
    def forward(self, x, c):
        c=c.view( c.shape[0], 1 )
        c=torch.tensor(c).float()        
        res={}
        mc_samples=50
        em, ev= self.encoder( x, c )
        res['em'] =em
        res['ev'] =ev
        res['z'] =[]
        res['x_pred'] =[]
        res['mc_samples']=mc_samples
        for i in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred= self.decoder( z, c )
            res['z'].append(z)
            res['x_pred'].append(x_pred)
    
        return res

    def compute_elbo(self, x, c, pred_model):
        c=torch.tensor(c).float()
        c=c.view( c.shape[0], 1 )    
        em, ev = self.encoder(x, c)
        kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 

        z = self.sample_latent_code(em, ev)
        dm= self.decoder(z, c)
        log_px_z = torch.tensor(0.0)
        
        x_pred= dm
        return torch.mean(log_px_z), torch.mean(kl_divergence), x, x_pred, torch.argmax( pred_model(x_pred), dim=1 )

class AutoEncoder(nn.Module):
    
    def __init__(self, data_size, encoded_size):

        super(AutoEncoder, self).__init__()
        
        self.data_size = data_size
        self.encoded_size= encoded_size
        # size= (W-Kernel_Size)/Stride + 1 after one conv2d; downsampling by factor of 2 in case of maxpool
        self.convoluted_size=int( (self.data_size - 3*2) ) 

        # No Categorical Variables for MINST, everything is continuous 
        self.encoded_categorical_feature_indexes= []     
        self.encoded_continuous_feature_indexes=[]
        self.encoded_start_cat= 0
        
        self.encoder_mean_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            )

        self.encoder_mean_fc = nn.Sequential(
            nn.Linear( self.convoluted_size**2, self.encoded_size),
            nn.BatchNorm1d(self.encoded_size),            
            nn.ReLU(),
            nn.Linear(self.encoded_size, self.encoded_size)
            )
        
        self.encoder_var_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            )
 
        self.encoder_var_fc = nn.Sequential(
            # Plus 1 to the input encoding size and data size to incorporate the target class label                    
            nn.Linear( self.convoluted_size**2, self.encoded_size),
            nn.BatchNorm1d(self.encoded_size),
            nn.ReLU(),
            nn.Linear(self.encoded_size, self.encoded_size),
            nn.Sigmoid(),
            )
 
        self.decoder_mean_fc = nn.Sequential(
            nn.Linear( self.encoded_size, self.convoluted_size**2 ),
            nn.BatchNorm1d(self.convoluted_size**2),
            nn.ReLU(),
            nn.Linear( self.convoluted_size**2, self.convoluted_size**2 ),
            nn.BatchNorm1d(self.convoluted_size**2),
            )

        self.decoder_mean_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size= 3, stride=1, padding=0 ),
            #nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(16, 16, kernel_size= 3, stride=1, padding=0 ),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid(),
            )
 
        
    def encoder(self, x):
        x= x.view(-1, 1, self.data_size, self.data_size)
        
        mean= self.encoder_mean_conv(x)
        mean= mean.view(-1, self.convoluted_size**2)
        mean= self.encoder_mean_fc(mean)
        
        logvar= self.encoder_var_conv(x)
        logvar= logvar.view( -1, self.convoluted_size**2) 
        logvar = 0.5+ self.encoder_var_fc(logvar)
        
        return mean, logvar

    def decoder(self, z):
        mean= self.decoder_mean_fc(z)
        mean= mean.view(-1, 1, self.convoluted_size, self.convoluted_size)
        mean= self.decoder_mean_conv(mean)
        mean= mean.view(-1, self.data_size**2)
        
        return mean
    
    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum( -.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar) ), axis=1)
                          
    def forward(self, x):        
        res={}
        mc_samples=50
        em, ev= self.encoder(x)
        res['em'] =em
        res['ev'] =ev
        res['z'] =[]
        res['x_pred'] =[]
        res['mc_samples']=mc_samples
        for i in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred= self.decoder(z)
            res['z'].append(z)
            res['x_pred'].append(x_pred)
    
        return res

