import os
import random
import time
import numpy as np
from PIL import Image, ImageOps
import numbers
import torch.nn.functional as F
from torchvision.transforms import Pad,RandomHorizontalFlip
from torchvision.transforms import ToTensor, ToPILImage


def _is_pil_image(img):
        return isinstance(img, Image.Image)

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    def __init__(self, size, padding=0, p=0.3):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.p = p

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):  #crop the same area of ori-image and label
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            if self.padding > 0:
                img = F.pad(img, self.padding)

            i, j, h, w = self.get_params(img, self.size)

            return crop(img, i, j, h, w)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomFlip(object):
    """Randomflip the given PIL Image randomly with a given probability. horizontal or vertical
    Args:
        p (float): probability of the image being flipped. Default value is 0.3
    """ 
        # make sure that crop area of  image and label are the same
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class CenterCrop(object):

    def __init__(self, size, padding=0, p=0.3):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.p = p
       
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            w, h = img.size
            th, tw = self.size
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            return crop(img, j, i, tw, th)
        else:
            return img


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
        self.p = 0.3 # change it to 100% during testing the robustness of the tirgger (inference phase)
        # self.p = 1

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_degree, Image.BILINEAR)
        else:
            return img

class RandomTrans(object):
    def __init__(self, degree=60):
        self.degree = degree
        self.p = 0.3

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_degree, Image.BILINEAR)
        else:
            if random.random() < self.p:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < self.p:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return img



class ReScalling(object):
    def __init__(self, ratio=2):
        self.ratio = ratio
        self.p = 0.3
        # self.p = 1

    def __call__(self, img):
        if random.random() < self.p:
            h, w = img.size
            random_W = random.randint(w//2, w*2)
            random_H = random.randint(h//2, h*2)

            return img.resize((random_H, random_W), Image.BILINEAR)
        else:
            return img

class RandMask(object):
    def __init__(self, ratio=0.25):
        self.ratio = ratio
        self.p = 0.3
        # self.p = 1

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            h, w,c = img.shape
            mask = np.random.choice(a=[0, 1], size=h*w, p=[self.ratio, 1 - self.ratio])
            mask = mask.reshape((h, w))
            for i in range(0, c):

                img[:,:,i] = img[:,:,i] * mask
            img = Image.fromarray(img)
        return img



class AddSaltPepperNoise(object):
    def __init__(self, density=0.1):
        self.density = density
    def __call__(self, img):
        img = np.array(img)
        h, w, c= img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(np.uint8(img))
        return img

class AddGussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = 1
    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = img + N
            img[img > 255] = 255
            img = Image.fromarray(np.uint8(img))
            return img
        else:
            return img


