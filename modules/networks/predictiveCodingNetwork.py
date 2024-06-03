import torch
import torch.nn as nn
from modules.networks.predictiveCodingLayer import PredictiveCodingLayer


class PredictiveCodingNetwork(nn.Module):

    def __init__(self, input_size, n_layers, n_causes, kernel_size=3, stride=1, padding=0, k1=0.2, k2=0.1, sigma2=0.1,
                 alpha=0.1, lam=1):
        super(PredictiveCodingNetwork, self).__init__()
        # input_size is the size of the input images (channels, height, width)
        # n_layers is the number of layers in the network
        # n_causes is the number of causes in each layer (can be a single number or a list of numbers)
        # kernel_size is the size of the patches (can be a single number or a list of numbers)
        # stride is the stride of the patches (can be a single number or a list of numbers)
        # padding is the padding of the patches (can be a single number or a list of numbers)
        # lam is the regularization parameter for the U prior (can be a single number or a list of numbers)
        # alpha is the regularization parameter for the r prior (can be a single number or a list of numbers)
        # k1 is the regularization parameter for the U prior (can be a single number or a list of numbers)
        # k2 is the regularization parameter for the r prior (can be a single number or a list of numbers)
        # sigma2 is the regularization parameter for the r prior (can be a single number or a list of numbers)

        # first, we need to make sure that the parameters are lists
        # if the parameters are single numbers, then we need to make them lists
        # lam, alpha, k1, k2, sigma2 could be floats
        # n_causes, kernel_size, stride, padding could be ints
        if type(n_causes) == int:
            n_causes = [n_causes] * n_layers
        if type(kernel_size) == int:
            kernel_size = [kernel_size] * n_layers
        if type(stride) == int:
            stride = [stride] * n_layers
        if type(padding) == int:
            padding = [padding] * n_layers
        if type(lam) == float:
            lam = [lam] * n_layers
        if type(alpha) == float:
            alpha = [alpha] * n_layers
        if type(k1) == float:
            k1 = [k1] * n_layers
        if type(k2) == float:
            k2 = [k2] * n_layers
        if type(sigma2) == float:
            sigma2 = [sigma2] * n_layers

        # now we need to make sure that the lists are the correct length
        assert len(n_causes) == n_layers
        assert len(kernel_size) == n_layers
        assert len(stride) == n_layers
        assert len(padding) == n_layers
        assert len(lam) == n_layers
        assert len(alpha) == n_layers
        assert len(k1) == n_layers
        assert len(k2) == n_layers
        assert len(sigma2) == n_layers

        # now we can create the layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # create the layer
            layer = PredictiveCodingLayer(input_size, n_causes=n_causes[i], kernel_size=kernel_size[i],
                                          stride=stride[i], padding=padding[i], k1=k1[i], k2=k2[i], sigma2=sigma2[i],
                                          alpha=alpha[i], lam=lam[i])

            # add the layer to the list of layers
            self.layers.append(layer)
            # update the input size. the input size for the next layer is the output size of the previous layer
            # which is the number of causes in the previous layer, the number of patches in the x direction, and the number of patches in the y direction
            input_size = (n_causes[i], layer.n_patches_height, layer.n_patches_width)

    def compute_total_loss(self):
        # compute the total loss by summing the total loss of each layer
        total_loss = 0
        # keep track of reconstruction loss, U prior loss, and r prior loss for printing
        U_loss, r_loss = 0, 0
        reconstruction_loss = 0
        mean_abs_error = 0

        for layer in self.layers:
            total_loss += layer.total_loss
            U_loss += layer.U_prior_loss
            r_loss += layer.r_prior_loss
            reconstruction_loss += layer.reconstruction_loss
            mean_abs_error += layer.mean_abs_error
        return total_loss, U_loss, r_loss, reconstruction_loss, mean_abs_error

    def forward_one_timestep(self, x):
        # x is the input image, of size (batch_size, channels, height, width)

        # run the network for one timestep
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x, timesteps=1, train_U=False, continuous=False, pbar=None):
        # x is the input image, of size (batch_size, channels, height, width)
        # timesteps is the number of timesteps to run the network for
        # r is the initial value of r, of size (batch_size, causes, patches_x, patches_y) for each layer

        # if its not continuous reset the representation array
        if not continuous:
            for layer in self.layers:
                # random but summing along n_causes dim should be 1, and all entries should be positive
                r = torch.rand(x.shape[0], layer.n_causes, layer.n_patches_height, layer.n_patches_width)
                r = r / r.sum(dim=1, keepdim=True)
                layer.set_r(r)

        # run the network for the specified number of timesteps, use tqdm to show a progress bar
        # and add the loss to the progress bar

        for i in range(timesteps):
            # run the network for one timestep
            _ = self.forward_one_timestep(x)

            # compute the total loss
            total_loss, U_loss, r_loss, reconstruction_loss, mean_abs_error = self.compute_total_loss()

            # print('timestep: {}, total_loss: {}'.format(i, total_loss))
            # add the total loss to the progress bar, as the reconstruction loss and mean absolute error
            # display with 3 decimal places
            # take gradients
            total_loss.backward()

            # compute the gradient of U and r
            # U_grads = [layer.U.grad for layer in self.layers]
            # r_grads = [layer.r.grad for layer in self.layers]

            # update U and r
            for layer in self.layers:
                # update r
                layer.r.data -= layer.k1 / 2 * layer.r.grad
                # update U
                if train_U:
                    layer.U.data -= layer.k2 / 2 * layer.U.grad

            # now zero the gradient of the model
            self.zero_grad()

        if pbar is not None:
            pbar.set_description(
                'total_loss: {:.3f}, reconstruction_loss: {:.3f}, mean_abs_error: {:.3f}'.format(total_loss,
                                                                                                 reconstruction_loss,
                                                                                                 mean_abs_error))
        return x
