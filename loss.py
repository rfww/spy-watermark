
import torch

def Get_loss_func(args):
    device = torch.device("cuda:0")
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():

        # criterion_GAN.cuda()
        # criterion_pixelwise.cuda()
        criterion_GAN.to(device)
        criterion_pixelwise.to(device)
    return criterion_GAN, criterion_pixelwise

