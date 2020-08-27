import torch
import math
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
from torch.nn import functional

class AttackLoss(torch.nn.Module):
    def __init__(self, target):
        super(AttackLoss, self).__init__()
        self.model = models.alexnet(pretrained=True).cuda()
        if target < 0:
            self.isTarget = False
            self.target_class = -target #target label in imagenet
        else:
            self.isTarget = True
            self.target_class = target #target label in imagenet
     
    def Transformations(self, img_torch):
        def Rotation(img_torch, angle):
            theta = torch.tensor([
                [math.cos(angle),math.sin(-angle),0],
                [math.sin(angle),math.cos(angle) ,0]
            ], dtype=torch.float)
            grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
            output = functional.grid_sample(img_torch.unsqueeze(0), grid)
            new_img_torch = output[0]
            return new_img_torch
        
        def Resize(img_torch, times):
            tmp = 1.0/times
            theta = torch.tensor([
                [tmp, 0  , 0],
                [0  , tmp, 0]
            ], dtype=torch.float)
            grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
            output = functional.grid_sample(img_torch.unsqueeze(0), grid)
            new_img_torch = output[0]
            return new_img_torch
        Img1 = Rotation(img_torch, -15*math.pi/180)
        Img2 = Resize(img_torch, 0.8)
        Img3 = Rotation(img_torch, 15*math.pi/180)
        TransImgs = torch.stack( (img_torch, img_torch, Img1,Img2,Img3) ,dim = 0)
        return TransImgs

    def forward(self, Images): 
        batch_size_cur = Images.shape[0]
        attackloss = 0.0
        for i in range(batch_size_cur):
            I = Images[i,:].squeeze()
            TransImg = self.Transformations(I)
            output = self.model(TransImg)
            print('max:' , (functional.softmax(output)).max(1))
            print('target:' ,functional.softmax(output)[:, self.target_class])
            if self.isTarget == True:
                attackloss -= (functional.softmax(output)[:, self.target_class]).mean()
            else: 
                attackloss += (functional.softmax(output)[:, self.target_class]).mean()
            print('loss:', attackloss)
        attackloss = attackloss/batch_size_cur
        return attackloss


class PercepLoss(torch.nn.Module):
    def __init__(self):
        super(PercepLoss, self).__init__()
        self.original_model = models.resnet50(pretrained=True)
        self.criterionPercep = torch.nn.MSELoss()
        self.features = torch.nn.Sequential(
            # stop at conv4
            *list(self.original_model.children())[:8]
        )
    def forward(self, Images_r, Images_f):
        batch_size_cur = Images_r.shape[0]
        perceploss = 0.0
        for i in range(batch_size_cur):
            I_r = Images_r[i,:].squeeze()
            I_f = Images_f[i,:].squeeze()
            features_r = self.features(I_r)
            features_f = self.features(I_f)
            perceploss += self.criterionPercep(features_r, features_f)
        return perceploss


class AttackPercepPix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_percep', type=float, default=0.0, help='weight for perceptural loss')
            parser.add_argument('--target', type=int, required=True, help='target index')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake','G_ATTACK', 'percep']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPercep = torch.nn.MSELoss()
            self.criterionATTACK = AttackLoss(opt.target)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        #ATTACK loss
        self.loss_G_ATTACK = self.criterionATTACK(self.fake_B)* self.opt.lambda_ATTACK
        # perceptual
        if self.opt.lambda_percep > 0:
            self.loss_percep = self.criterionPercep(self.real_A, self.fake_B) * self.opt.lambda_percep
        else: 
            self.loss_percep = 0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_percep
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
