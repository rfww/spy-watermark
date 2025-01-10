import torch
from models.models import Create_nets
from datasets import *
from options import TrainOptions
from optimizer import *
from utils import *
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
#load the args
args = TrainOptions().parse()

# Initialize generator
generator = Create_nets(args)

# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
criterion_cross = nn.CrossEntropyLoss().cuda()
mask_loss_fn = MaskLoss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)
# Optimizers
optimizer_G = Get_optimizers(args, generator)
log={'bestmse_it':0, 'best_mse':100, 'psnr':0, 'bestpsnr_it':0, 'best_psnr':0, 'mse':0}
f = open(os.path.join(args.model_result_dir, "training.txt"), 'w')
f.write("***************************Training***************************\n")
# Configure dataloaders
real_loder = Get_dataloader(args.path+"/train", args.batch_size, args.img_size)
test_loder = Get_dataloader_test(args.path+"/val", 1, args.img_size, shuffle=True)
# real_loder2 = Get_dataloader(args.path+"/n01944390", args.batch_size)

real = iter(real_loder)

j=0
# 开始训练

pbar = range(args.epoch_start, 100000)
for i in pbar:
    try:
        image, mark = next(real)
        image = image.to(device)
        mark = mark.to(device)
    except (OSError, StopIteration):
        real = iter(real_loder)
        image, mark = next(real)
        image = image.to(device)
        mark = mark.to(device)



    # ------------------
    #  Train Generators
    # ------------------

    # Adversarial ground truths
    patch = (1, 1, 1)
    valid = Variable(torch.FloatTensor(np.ones((image.size(0)))).cuda(), requires_grad=False)
    fake  = Variable(torch.FloatTensor(np.zeros((image.size(0)))).cuda(), requires_grad=False)
    label = Variable(image, requires_grad=False)
    markl = Variable(mark, requires_grad=False)
    markz = torch.zeros_like(markl)
    optimizer_G.zero_grad()
    requires_grad(generator, True)


    loss_G, pred, mark1, mark2, m1_2, m1_3, m2_2, m2_3 = generator(image, mark)
    loss1 = criterion_pixelwise(mark1, markz)
    loss1_2 = criterion_pixelwise(m1_2, markz)
    loss1_3 = criterion_pixelwise(m1_3, markz)
    loss2 = criterion_pixelwise(mark2, markl)
    loss2_2 = criterion_pixelwise(m2_2, markl)
    loss2_3 = criterion_pixelwise(m2_3, markl)
    loss_G = loss_G + (loss1 + loss1_2+loss1_3)+10*(loss2+loss2_2+loss2_3)  # CIF10_E1_C1
    loss_G.backward()
    optimizer_G.step()


    if i%100==0:
        print("\r[Batch%d]-[loss_G:%f]-[loss_C:0f]" %(i, loss_G.data.cpu()))

    if args.checkpoint_interval != -1 and i != 0 and i % 1000 == 0:
        image_path = '%s/training' % (args.model_result_dir)
        os.makedirs(image_path, exist_ok=True)
        to_image(image, i=i, tag='input', path=image_path)
        to_image(pred, i=i, tag='recon', path=image_path)
        to_image_mask(mark1, i=i, tag='mark1', path=image_path)
        to_image_mask(mark2, i=i, tag='mark2', path=image_path)
        # Save model checkpoints
        torch.save(generator.state_dict(), '%s/generator_latest.pth' % (args.model_result_dir))
        generator.eval()
        psnr = 0
        mse = 0
        for j, (img, mark, _) in enumerate(tqdm(test_loder, total=1000)):
            img = img.cuda()
            mark = mark.cuda()
            _, pred, mark1, mark2, m1_2, m1_3, m2_2, m2_3 = generator(img, mark)

            rec = torch.einsum('nchw->nhwc', pred).detach().cpu()
            img = torch.einsum('nchw->nhwc', img).detach().cpu()
            rec = torch.clip((rec[0] * 0.5 + 0.5) * 255, 0, 255).int().numpy()
            img = torch.clip((img[0] * 0.5 + 0.5) * 255, 0, 255).int().numpy()
            psnr += PSNR(rec, img)
            markl = mark.detach().cpu().numpy()
            markz = torch.zeros_like(mark).cpu().numpy()
            mark1 = mark1.detach().cpu().numpy()
            mark2 = mark2.detach().cpu().numpy()
            mse += np.mean(np.abs((mark1 - markz)) ** 2) + np.mean(np.abs((mark2 - markl)) ** 2)
            if j + 1 == 1000:
                break

        psnr /= 1000.0
        mse /= 1000.0
        if mse < log['best_mse']:
            log['bestmse_it'] = i
            log['best_mse'] = mse
            log['psnr'] = psnr
            torch.save(generator.state_dict(), '%s/generator_Mbest.pth' % (args.model_result_dir))
        if psnr > log['best_psnr']:
            log['bestpsnr_it'] = i
            log['best_psnr'] = psnr
            log['mse'] = mse
            torch.save(generator.state_dict(), '%s/generator_Gbest.pth' % (args.model_result_dir))
        print('=======================================================================================================')
        print('batch:', i, "mse:", mse, "psnr:", psnr)
        print('bestmse_it', log['bestmse_it'], 'best_mse', log['best_mse'], 'psnr:', log['psnr'])
        print('bestpsnr_it', log['bestpsnr_it'], 'mse:', log['mse'], 'best_psnr', log['best_psnr'])
        print('=======================================================================================================')
        f.write("batch: %d, mse: %f, psnr: %f\n" % (i, mse, psnr))
        f.write("bestmse_it: %d, best_mse: %f, psnr: %f\n" % (log['bestmse_it'], log['best_mse'], log['psnr']))
        f.write("bestpsnr_it: %d, mse: %f, best_psnr: %f\n" % (log['bestpsnr_it'], log['mse'], log['best_psnr']))
        f.flush()

f.close()
