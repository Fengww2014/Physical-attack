import torch
import math
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import torchvision.transforms as transforms
from torch.nn import functional
from torch.autograd import Variable
import numpy as np

class AttackLoss(torch.nn.Module):
    def __init__(self, target, ori):
        super(AttackLoss, self).__init__()
        self.model1 = models.vgg16(pretrained=True).cuda()
        if target < 0:
            self.isTarget = False
            self.target_class = -target #target label in imagenet
        else:
            self.isTarget = True
            self.target_class = target #target label in imagenet
            self.ori = ori
     
     
    def Transformations(self, img_torch):
        def Rotation(img_torch, angle):
            theta = torch.tensor([
                [math.cos(angle),math.sin(-angle),0],
                [math.sin(angle),math.cos(angle) ,0]
            ], dtype=torch.float) + (torch.rand(2,3)-0.5)/5
            grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
            output = functional.grid_sample(img_torch.unsqueeze(0), grid)
            new_img_torch = output[0]
            return new_img_torch
        
        def Resize(img_torch, times):
            tmp = 1.0/times
            theta = torch.tensor([
                [tmp, 0  , 0],
                [0  , tmp, 0]
            ], dtype=torch.float) + (torch.rand(2,3)-0.5)/5
            grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
            output = functional.grid_sample(img_torch.unsqueeze(0), grid)
            new_img_torch = output[0]
            return new_img_torch
        Img0 = img_torch + torch.randn_like(img_torch) * 0.1
        Img1 = Resize(img_torch+torch.randn_like(img_torch) * 0.1, 0.8+(np.random.rand()-0.5)*0.1) 
        Img2 = Resize(img_torch+torch.randn_like(img_torch) * 0.1, 1.2+(np.random.rand()-0.5)*0.1)
        Img3 = Resize(img_torch+torch.randn_like(img_torch) * 0.1, 0.6+(np.random.rand()-0.5)*0.1) 
        Img28 = Rotation(img_torch, (np.random.rand()-0.5)*10*math.pi/180)
        Img4 = Rotation(img_torch, (10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img5 = Rotation(img_torch, -(10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img6 = Rotation(img_torch, (20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img7 = Rotation(img_torch, -(20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img29 = Rotation(Img1, (np.random.rand()-0.5)*10*math.pi/180)
        Img8 = Rotation(Img1, (10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img9 = Rotation(Img1, -(10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img10 = Rotation(Img1, (20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img11 = Rotation(Img1, -(20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img30 = Rotation(Img2, (np.random.rand()-0.5)*10*math.pi/180)
        Img12 = Rotation(Img2, (10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img13 = Rotation(Img2, -(10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img14 = Rotation(Img2, (20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img15 = Rotation(Img2, -(20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img31 = Rotation(Img3, (np.random.rand()-0.5)*10*math.pi/180)
        Img16 = Rotation(Img3, (10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img17 = Rotation(Img3, -(10+(np.random.rand()-0.5)*10)*math.pi/180)
        Img18 = Rotation(Img3, (20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img19 = Rotation(Img3, -(20+(np.random.rand()-0.5)*10)*math.pi/180)
        Img20 = Rotation(img_torch, (30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img21 = Rotation(img_torch, -(30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img22 = Rotation(Img1, (30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img23 = Rotation(Img1, -(30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img24 = Rotation(Img2, (30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img25 = Rotation(Img2, -(30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img26 = Rotation(Img3, (30+(np.random.rand()-0.5)*10)*math.pi/180)
        Img27 = Rotation(Img3, -(30+(np.random.rand()-0.5)*10)*math.pi/180)
        TransImgs = torch.stack((img_torch,img_torch,img_torch,Img0,Img0,Img1,Img1,Img2,Img2,Img3,Img3,Img4,Img5,Img6,Img7,Img8,Img9,Img10,Img11,Img12,Img13,Img14,Img15,Img16,Img17,Img18,Img19,Img20,Img21,Img22,Img23,Img24,Img25,Img26,Img27, Img28, Img29, Img30, Img31),dim = 0)
        return TransImgs

    def forward(self, Images): 
        batch_size_cur = Images.shape[0]
        attackloss1 = 0.0
        for i in range(batch_size_cur):
            I = Images[i,:].squeeze()
            TransImg = self.Transformations(I)
            output1 = self.model1(TransImg)
            #print('max:' , (functional.softmax(output)).max(1))
            #print('target:' ,functional.softmax(output)[:, self.target_class])
            if self.isTarget == True:
                attackloss1 = attackloss1 - (functional.softmax(output1)[:, self.target_class]).mean() + (functional.softmax(output1)[:, self.ori]).mean()
            else: 
                attackloss1 += (functional.softmax(output1)[:, self.target_class]).mean()
            #print('loss:', attackloss)
        attackloss = attackloss1/batch_size_cur
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


class AttackEtTarGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_distance', type=float, default=0.0, help='weight for distance loss')
            parser.add_argument('--target', type=int, required=True, help='target index')
            parser.add_argument('--lambda_percep', type=float, default=0.0, help='weight for perceptural loss')
            parser.add_argument('--ori', type=int, default=0, help='ori label')
            parser.add_argument('--lambda_ATTACK_B', type=float, default=0.0, help='weight for ATTACK loss for adversarial attack in fake B')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'idt_A', 'G_ATTACK', 'distance']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_A')

        self.visual_names = visual_names_A  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A',  'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionATTACK = AttackLoss(opt.target, opt.ori)
            self.criterionDis = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_B = self.opt.lambda_B
        lambda_att_B = self.opt.lambda_ATTACK_B
        lambda_distance = self.opt.lambda_distance
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        #ATTACK loss
        if lambda_att_B > 0:
            self.loss_G_ATTACK = self.criterionATTACK(self.fake_B)* lambda_att_B
        else:
            self.loss_G_ATTACK = 0
        if lambda_distance > 0:
            self.loss_distance = self.criterionDis(self.real_A, self.fake_B) * lambda_distance
        else: 
            self.loss_distance = 0
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_idt_A + self.loss_G_ATTACK - self.loss_distance
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD_A, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights


