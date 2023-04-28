import torch.nn as nn
import torch

from models.Extractor import GeneratorUNet

def Create_nets(args):
    generator = GeneratorUNet(args.img_size)
    if torch.cuda.is_available():
        generator = generator.cuda()

    if args.resume is not None:
        # Load pretrained models
        generator.load_state_dict(torch.load(args.resume))

    return generator

if __name__ == '__main__':
    # c=Discriminator()
    print()
