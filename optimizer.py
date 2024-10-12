import torch
# Optimizers
def Get_optimizers(args, generator):

    optimizer_G = torch.optim.SGD(
        generator.parameters(),
        lr=args.lr, momentum=0.5)
    return optimizer_G

# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.MSELoss()
    # criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    return criterion_GAN, criterion_pixelwise

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
