import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import math



def to_image(tensor,i,tag,path):
    #for i in range(32):
    image = tensor
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path+'/{}.png'.format(str(i)+'_'+tag)
    save_image(image.detach(),
               fake_samples_file,
               normalize=True,
               range=(-1., 1.),
               nrow=4)

#保留mask
def to_image_mask(tensor, i,tag, path):
    image = tensor  # [i].cpu().clone()
    if not os.path.isdir(path):
        os.makedirs(path)
    fake_samples_file = path + '/{}.png'.format(str(i)+'_'+tag)
    save_image(image.detach(),
               fake_samples_file,
               normalize=True,
               range=(0., 1.),
               nrow=4)

def to_image_test(tensor, name, path):
    # mask = tensor.detach().cpu().numpy()[0,0,:,:]  # [i].cpu().clone()
    mask = tensor.detach().cpu().numpy()[0,:,:,:]  # [i].cpu().clone()
    mask = mask.transpose(1, 2, 0)
    # print(mask.shape)
    if not os.path.isdir(path):
        os.makedirs(path)
    # fake_samples_file = path + '/{}.bmp'.format(str(i))
    fake_samples_file = path + '/{}'.format(name)
    # mask=Image.fromarray(mask*255).convert('L')
    mask=Image.fromarray(np.uint8(mask*255))
    mask.save(fake_samples_file)
    # save_image(image.detach(),
    #            fake_samples_file,
    #            normalize=True,
    #            range=(0., 1.),
    #            nrow=4)

def PSNR(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)











