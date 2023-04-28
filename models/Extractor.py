import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import torch
import numpy as np
# import cupy as np
import math
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torchvision.transforms as transforms
from .pos_embed import get_2d_sincos_pos_embed



class GeneratorUNet(nn.Module):
    def __init__(self, img_size=224):
        super(GeneratorUNet, self).__init__()
        # self.resnet_DE = resnet_d1_e1()
        self.img_size = (img_size, img_size)
        self.base2 = Base2()
        self.vit = Autoencoder(img_size)
        # self.height = (256, 1080)
        # self.width = (256, 1920)
        # self.trans = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def forward(self, input_1, mark=None):
        # db= self.resnet_DE(input_1, mark)
        loss, pred = self.vit(input_1, mark)
        #---------------------------------------
        # random scale
        # randH = np.random.randint(self.height)
        # randW = np.random.randint(self.width)

        # if np.random.random() < 0.5:
        #     if np.random.random() < 0.5:
        #         randH = np.random.randint(32, 128)
        #         randW = np.random.randint(32, 128)
        #         input_1 = F.upsample_bilinear(input_1, size=(randW, randH))
        #         pred = F.upsample_bilinear(pred, size=(randW, randH))
        #     else:
        #         prob = 0.25
        #         mask = np.random.choice(a=[0, 1], size=self.img_size, p=[prob, 1 - prob])
        #         mask = torch.from_numpy(mask).cuda()
        #         input_1 = input_1 * mask
        #         pred = pred * mask
        #
        # input_1 = F.upsample_bilinear(input_1, size=self.img_size)
        # pred = F.upsample_bilinear(pred, size=self.img_size)


        #---------------------------------------
        mark1, m1_2, m1_3 = self.base2(input_1)
        mark2, m2_2, m2_3 = self.base2(pred)
        # torch.save(self.base2.state_dict(), "ISIC_base2.pth")
        return loss, pred, mark1, mark2, m1_2, m1_3, m2_2, m2_3
        # return None, None, mark1, None
        # return pred



class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input





class Base2(nn.Module):
    def __init__(self, eps=0.3):
        super(Base2, self).__init__()
        self.eps = eps
        self.maxpool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_3 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        # self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_1 = BaseConv(768, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv6_1 = BaseConv(384, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_3 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv7_1 = BaseConv(128, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_3 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        # self.conv8_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv8_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        #
        # self.conv9_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_4 = BaseConv(256, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv6_4 = BaseConv(64, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv7_4 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)



    def forward(self, x, is_noise=False):

        if is_noise:
            random_noise = torch.empty(x.shape).uniform_().cuda()
            random_noise = torch.mul(torch.sign(x), F.normalize(random_noise, p=2, dim=1)) * self.eps
            #print(random_noise)
            x = x+ random_noise

        x = self.conv1_1(x)
        x1 = self.conv1_2(x)

        x = self.maxpool(x1)  #
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x2 = self.conv2_3(x)

        x = self.maxpool(x2)  #
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x3 = self.conv3_3(x)

        x = self.maxpool(x3)  #
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x4 = self.conv4_3(x)

        x = self.upsample(x4)  #
        x = self.conv5_1(torch.cat([x3, x], 1))
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        o5 = self.conv5_4(x)

        x = self.upsample(x)
        x = self.conv6_1(torch.cat([x2, x], 1))
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        o6 = self.conv6_4(x)

        x = self.upsample(x)
        x = self.conv7_1(torch.cat([x1, x], 1))
        x = self.conv7_2(x)
        x = self.conv7_3(x)

        x = self.conv7_4(x)

        o5 = F.upsample_bilinear(o5, scale_factor=4)
        o6 = F.upsample_bilinear(o6, scale_factor=2)
        return x, o5, o6






class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def forward(self, x, H, W):
    def forward(self, x):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class Autoencoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_mark = nn.Linear(256, 1024)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
   
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.mlp = Mlp(decoder_embed_dim, decoder_embed_dim // 2, decoder_embed_dim)
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def patch_mark(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


    def forward_decoder(self, x, mark):
        # embed tokens
        x = self.decoder_embed(x)
        mark = self.patch_mark(mark)
        mark = self.mask_mark(mark)


        # add pos embed

        x = x + mark
        x = self.mlp(x)


        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        return x


    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5



        loss = (F.relu(abs(pred - target ) - 1 /255.)) ** 2
        # loss = (pred - target) ** 2

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.sum()
        return loss

    def forward(self, imgs, mark):

        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, mark)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)
        return loss, self.unpatchify(pred)
        # return self.unpatchify(pred)
