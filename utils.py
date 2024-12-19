import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import math


def sample_images(generator, test_dataloader, args, epoch, batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs['A'].type(torch.FloatTensor).cuda())
    real_B = Variable(imgs['B'].type(torch.FloatTensor).cuda())
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, '%s-%s/%s/%s-%s.png' % (args.exp_name, args.dataset_name, args.img_result_dir, batches_done, epoch), nrow=5, normalize=True)

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











