import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def patches2image_batch(x, output_size):
    B, C, N_PATCH, H_PATCH, W_PATCH = x.shape
    x = x.reshape(B, C, N_PATCH, -1).transpose(2, 3).reshape(B, -1, N_PATCH)  # [B, C, prod(PATCH_SIZE)]
    x = F.fold(x, output_size, kernel_size=(H_PATCH, W_PATCH), stride=(H_PATCH, W_PATCH))  # [B, C, H, W]
    return x  # [B, C, H, W]


def attention(x):
    """
    Inputs:
        x: tensor of shape [B, N, D]
    Outputs:
        x: tensor of shape [B, N, D]
    """
    att = torch.bmm(x, x.transpose(1, 2))
    att = F.softmax(att, 1)
    x = torch.bmm(att, x)
    return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=(1, 16), in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BDHW -> BND
        x = self.norm(x)
        return x


def random_blackout(tensor, percentage=0.1):
    """
    Sets a given percentage of pixels in a tensor to black (zero) using NumPy.

    Args:
    tensor (np.ndarray): The input tensor with shape [B, C, H, W].
    percentage (float): The percentage of pixels to set to black (default is 0.05, i.e., 5%).

    Returns:
    np.ndarray: The tensor with randomly blacked out pixels.
    """
    # Get the shape of the tensor
    B, C, H, W = tensor.shape

    # Calculate the total number of pixels
    total_pixels = B * H * W

    # Calculate the number of pixels to black out
    num_pixels_to_blackout = int(total_pixels * percentage)

    # Generate random indices for the pixels to blackout
    indices = np.random.choice(total_pixels, num_pixels_to_blackout, replace=False)

    # Convert the 1D indices to 3D indices (batch, height, width)
    batch_indices = indices // (H * W)
    hw_indices = indices % (H * W)
    height_indices = hw_indices // W
    width_indices = hw_indices % W

    # Set the selected pixels to black (zero) across all channels
    tensor[batch_indices, :, height_indices, width_indices] = 0

    return tensor
