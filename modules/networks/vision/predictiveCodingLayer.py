import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum


# Define the model
# Define the model
class PredictiveCodingLayer(nn.Module):

    def __init__(self, input_size, n_causes=32, kernel_size=16, stride=5, padding=0,
                 k1=0.1, k2=0.1, sigma2=1, alpha=0.1, lam=0.1, f=nn.Tanh(), U_prior=None, r_prior=None):
        '''
        This is the initialization function for a predictive coding layer.

        input_size: the size of the full input image (channels, height, width)
        n_causes: the number of causes we use as a basis to predict the image
        kernel_size: the size of the patches, can be a single integer or a tuple (height, width)
        stride: the stride of the patches, can be a single integer or a tuple (height, width)
        padding: the padding of the patches, can be a single integer or a tuple (height, width)
        k1: the learning rate for r
        k2: the learning rate for U
        sigma2: the variance of the noise
        alpha: weight on the r prior
        lam: weight on the U prior
        f: the nonlinearity to use
        U_prior: the prior on U, if None, use the default prior
        r_prior: the prior on r, if None, use the default prior
        '''
        super(PredictiveCodingLayer, self).__init__()

        self.input_size = input_size # (channels, height, width)
        self.n_causes = n_causes

        # process the patch parameters
        self.kernel_size, self.stride, self.padding = self.process_patch_params(kernel_size, stride, padding)

        # create the unfold layer, this will be used to extract patches from the image
        self.unfold = nn.Unfold(self.kernel_size, stride=self.stride, padding=self.padding)
        self.fold = nn.Fold(self.input_size[1:], self.kernel_size, stride=self.stride, padding=self.padding)
        self.n_patches = self.unfold(torch.zeros(1, *input_size)).shape[2]
        self.n_patches_height, self. n_patches_width = self.compute_patch_shape()

        # set the learning rates and other parameters
        self.k1 = k1 # the learning rate for r
        self.k2 = k2 # the learning rate for U
        self.sigma2 = sigma2 # the variance of the noise
        self.alpha = alpha # weight on the r prior
        self.lam = lam # weight on the U prior
        self.precision = 1/sigma2

        self.f = f

        # set the priors on r and U, defaults are L2 loss
        if U_prior is None:
            self.U_prior = lambda x: F.mse_loss(x, torch.zeros_like(x))
        else:
            self.U_prior = U_prior

        if r_prior is None:
            self.r_prior = lambda x: F.mse_loss(x, torch.zeros_like(x))
        else:
            self.r_prior = r_prior

        # initialize the losses
        self.reconstruction_loss = 0
        self.r_prior_loss = 0
        self.U_prior_loss = 0
        self.total_loss = 0
        self.mean_abs_error = 0

        # initialize the causes and the activations
        self.U = nn.Parameter(torch.randn(n_causes, input_size[0], self.kernel_size[0], self.kernel_size[1]))
        self.r = nn.Parameter(torch.randn(1, n_causes, self.n_patches_height, self.n_patches_width)) # (batch_size, n_causes, n_patches_height, n_patches_width)

        # we initialize the batch size to 1, but it will change when we get a new image
        self.batch_size = 1

    def process_patch_params(self, kernel_size, stride, padding):
        # if these are single integers, then we need to make them tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        return kernel_size, stride, padding

    def compute_patch_shape(self):
        '''
        This function computes the shape of the patches
        '''
        # calculate the number of patches in the height and width
        # based on the formula from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # (spatial_size[d]+2×padding[d]−dilation[d]×(kernel_size[d]−1)−1)/stride[d]+1
        # if padding, kernel_size or stride are ints then they are used for both height and width
        n_patches_height = (self.input_size[1]+2*self.padding[0]-(self.kernel_size[0]-1)-1)/self.stride[0]+1
        n_patches_width = (self.input_size[2]+2*self.padding[1]-(self.kernel_size[1]-1)-1)/self.stride[1]+1

        # take the floor
        n_patches_height = int(n_patches_height)
        n_patches_width = int(n_patches_width)

        # make sure that n_patches_height and n_patches_width are integers and they multipy to n_patches
        assert n_patches_height*n_patches_width == self.n_patches

        return n_patches_height, n_patches_width

    def set_r(self, r):
        '''
        This function sets the value of r
        r is the state of the causes (batch_size, causes, n_patches_height, n_patches_width)
        '''
        self.r = nn.Parameter(r)

    def forward(self, x):
        '''
        This is the forward function for the model.
        x is an image of size (batch_size, channels, height, width)
        r is the state of the causes (batch_size, causes, n_patches_height, n_patches_width)
        '''
        # set the batch size
        self.batch_size = x.shape[0]

        # first, get the image patches
        patches = self.unfold(x) # (batch_size, channels*kernel_size*kernel_size, num_patches)
        patches = patches.view(self.batch_size, self.input_size[0], self.kernel_size[0], self.kernel_size[1], self.n_patches_height, self.n_patches_width)
        # patches shape is (batch_size, channels, kernel_size_height, kernel_size_width, n_patches_height, n_patches_width)

        # I want to make a prediction for every patch, so I need to combine U by r
        # U is (causes, channels, kernel_size_height, kernel_size_width)
        # r is (batch_size, causes, n_patches_height, n_patches_width)
        # I want the output to be (batch_size, channels, kernel_size_height, kernel_size_width, n_patches_height, n_patches_width)
        prediction = einsum('causes chan kernh kernw, batch causes npatchesh npatchesw -> batch chan kernh kernw npatchesh npatchesw', self.U, self.r)
        prediction = self.f(prediction)

        # calculate the reconstruction loss
        prediction_error = patches - prediction
        self.reconstruction_loss = self.precision*torch.norm(prediction_error)**2
        # mean absolute error
        self.mean_abs_error = torch.mean(torch.abs(prediction_error))

        # calculate the prior losses
        self.U_prior_loss = self.lam*self.U_prior(self.U)
        self.r_prior_loss = self.alpha*self.r_prior(self.r)

        # calculate the total loss
        self.total_loss = self.reconstruction_loss
        self.total_loss += self.U_prior_loss
        self.total_loss += self.r_prior_loss

        # reshape self.r to be (batch_size, causes, patches_x, patches_y), to send to next level
        self.r.data = self.r.view(self.batch_size, self.n_causes, self.n_patches_height, self.n_patches_width)

        return self.r