#%% load cfar images, download through torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# load the dataset
from modules.networks.vision.glom.GLOM import GlomAE, GlomAEConfig

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True)

# get an image and plot it
dataiter = iter(trainloader)
images, labels = next(dataiter)
B, C, W, H = images.shape

# plot an image
#plt.imshow(np.transpose(images[0], (1, 2, 0)))

visConfig = GlomAEConfig(
    stereoscopic=False,
    img_size=(H, W),
    patch_size=(1, 4), n_embed=108, n_head=6,
    n_layers=3, in_chans=3,
    lr=0.01, wd=0.0001, betas=(0.9, 0.95),
)

model = GlomAE(visConfig)

for i in range(10):
    images, labels = next(dataiter)
    latent, rec, loss = model.backward(images.to("cuda"))
    print(loss)
