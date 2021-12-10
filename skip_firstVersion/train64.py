import os

import numpy as np
#import utils.util.gaussian as gaussian
from utils.util import gaussian as gaussian

from collections import OrderedDict
import cv2
import time
import json
from tqdm import tqdm

import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.optim as optim

from model.network.TraNetwork import defineG, defineD, defineDL
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

from utils.util import roc_plot

LAMBDA = 10.0 #梯度惩罚因子

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1).cuda(non_blocking=True)
    n, c, h, w = real_data.size()
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, h, w)
    # alpha = alpha.cuda(gpu) if use_cuda else alpha
    if torch.cuda.is_available():
        alpha = alpha.cuda(non_blocking=True)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda(non_blocking=True)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(non_blocking=True) if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples

class Train():
    def __init__(self, opt, train_data=None, test_data=None,test_label=None):
        super(Train, self).__init__()
        # 初始化随机种子
        self.opt = opt
        self.seed(self.opt.seed_value)

        # basic setting
        self.epoches = self.opt.niter
        self.bs = self.opt.batchsize
        self.sigma_noise = torch.tensor(self.opt.sigma_noise).cuda(non_blocking=True)
        self.image_grids_numbers = self.opt.image_grids_numbers
        self.n_row_in_grid = self.opt.n_row_in_grid
        self.output = self.opt.outf
        self.print_freq = self.opt.print_freq

        # training setting
        self.loss_type = self.opt.loss_type
        self.optim_type = self.opt.optim_type
        self.norm_type = "batch"
        self.init_type = "normal"

        self.train_data = train_data
        self.test_data = test_data
        self.name = self.opt.name
        self.real_label = test_label
    
        # 
        # weight path
        self.weight_dir = os.path.join(self.opt.outf, "ckpt", self.opt.name, self.opt.loss_type, self.opt.optim_type)
        self.trainImg_dir = os.path.join(self.opt.outf, "train/Images", self.opt.name, self.opt.loss_type,
                                         self.opt.optim_type)
        self.test_dir = os.path.join(self.opt.outf, "test/Images", self.opt.name, self.opt.loss_type,
                                         self.opt.optim_type)
        self.loss_dir = os.path.join(self.opt.outf, "train/loss", self.opt.name, self.opt.loss_type,
                                     self.opt.optim_type)

        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # loss function
        if self.loss_type == "l1":
            self.l_con = nn.L1Loss()
        elif self.loss_type == "mse":
            self.l_con = nn.MSELoss()
        elif self.loss_type == "ssim":
            self.l_con = SSIM()

        self.l_adv = nn.BCELoss()
        self.l_lat = nn.BCELoss()

        self.mse_loss = nn.MSELoss()
        # 权重参数
        self.w_adv = self.opt.w_adv
        self.w_con = self.opt.w_con
        self.w_lat = self.opt.w_lat

        self.netsize = self.opt.netsize


        # network
        self.G = defineG(self.opt, self.norm_type, self.init_type)
        self.D = defineD(self.opt, self.norm_type, self.init_type)
        self.DL = defineDL(self.opt, self.init_type)        
        # 优化器
        self.optim_d = optim.Adam(self.D.parameters(), lr=self.opt.d_lr, betas=(self.opt.beta1, 0.999))
        self.optim_g = optim.Adam(self.G.parameters(), lr=self.opt.g_lr, betas=(self.opt.beta1, 0.999))
        self.optim_DL = optim.Adam(self.DL.parameters(), lr=self.opt.d_lat_lr)
        
        # 网络加载cuda
        if torch.cuda.is_available():
            self.G = nn.DataParallel(self.G.cuda())
            self.D = nn.DataParallel(self.D.cuda())
            self.DL = nn.DataParallel(self.DL.cuda())
            self.optim_d = nn.DataParallel(self.optim_d)
            self.optim_g = nn.DataParallel(self.optim_g)
            self.optim_DL = nn.DataParallel(self.optim_DL)
            
        self.errors = {}

        # train label
        if torch.cuda.is_available():
            self.d_real_label = torch.tensor(np.random.uniform(0.7, 1.2, self.bs).astype("float32")).cuda(non_blocking=True)
            self.d_fake_label = torch.tensor(np.random.uniform(0., 0.3, self.opt.batchsize).astype("float32")).cuda(non_blocking=True)
            
            self.g_real_label = torch.ones(size=(self.bs, 1), dtype=torch.float32).cuda(non_blocking=True)
            #self.g_fake_label = torch.zeros(size=(self.bs, 1).astype("float32")).cuda(non_blocking=True)
            
            # latent limit
            self.real_z_label = torch.ones(size=(self.bs, 1), dtype=torch.float32).cuda(non_blocking=True)
            self.fake_z_label = torch.zeros(size=(self.bs, 1), dtype=torch.float32).cuda(non_blocking=True)
            self.real_z_data = torch.randn(self.bs, self.opt.lat_dim).cuda(non_blocking=True)
            self.real_z_data.requires_grad_(True)
        # random seed
    def seed(self, seed_value):
        """ Seed
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.randn(seed_value)
        torch.backends.cudnn.deterministic = True


    def freezeG(self):
        for g_grad in self.G.parameters():
            g_grad.requires_grad = False
        
        for d_grad in self.D.parameters():
            d_grad.requires_grad = True
        
        for dl_grad in self.DL.parameters():
            dl_grad.requires_grad = True

    def freezeD(self):
        for g_grad in self.G.parameters():
            g_grad.requires_grad = True
        for d_grad in self.D.parameters():
            d_grad.requires_grad = False
        for dl_grad in self.DL.parameters():
            dl_grad.requires_grad = False

    def freeDLUpdataD(self):
        for d in self.D.parameters():
            d.requires_grad = True
        for dl_grad in self.DL.parameters():
            dl_grad.requires_grad = False

    def freeDUpdataDL(self):
        for d_grad in self.D.parameters():
            d_grad.requires_grad = False
        for dl_grad in self.DL.parameters():
            dl_grad.requires_grad = True

    def train(self):
        for epoch in tqdm(range(self.epoches)):

            self.freezeG()
            for i, data in enumerate(self.train_data):
                self.input, _ = data
                if torch.cuda.is_available():
                    self.input = self.input.cuda(non_blocking=True)
                
                # 优化器清零
                # self.optim_g.zero_grad()
                self.optim_d.zero_grad()
                self.optim_DL.zero_grad()

                sigma = self.sigma_noise ** 2
                self.Input_with_noise = gaussian(self.input, True, 0, sigma).cuda(non_blocking=True)
                #self.Input_with_noise =self.Input_with_noise.cuda(non_blocking=True)

                ##############################
                #update netDL
                ##############################
                fake_img, fake_noise = self.G(self.Input_with_noise)
                #self.freeDUpdataDL()
                #fake_img, fake_noise = self.G(self.input)
                z_err = 0.5 *(self.l_lat(self.DL(self.real_z_data), self.real_z_label) + self.l_lat(self.DL(torch.squeeze(fake_noise.detach())),self.fake_z_label))
                z_err.backward()
                self.optim_DL.module.step()

                #################################
                #update D
                #################################
                #self.freeDLUpdataD()
                
                d_fake_score, d_fake_feature = self.D(fake_img)
                d_real_score, d_real_feature = self.D(self.input)
                d_fake_err = self.l_adv(torch.squeeze(d_fake_score), self.d_fake_label)
                d_real_err = self.l_adv(torch.squeeze(d_real_score), self.d_real_label)
                d_feature_err = self.mse_loss(d_real_feature, d_fake_feature)
                d_err = 0.5 * (d_fake_err + d_real_err) + d_feature_err

                d_err.backward()
                self.optim_d.module.step()

                # 添加损失
                self.add_loss(d_err, "d_err")
                self.add_loss(d_real_err, "d_real_err")
                self.add_loss(d_fake_err, "d_fake_err")
                self.add_loss(d_feature_err,  "d_reature_err")

                del d_fake_feature
                del d_real_feature
                del fake_img
                del fake_noise

                """
                # 这部分是需要加梯度惩罚项
                d_real_err = -torch.mean(self.D(self.input))
                d_fake_err = torch.mean(self.D(fake_img.detach()))
                d_err = d_fake_err + d_real_err
                # gradient penalty
                gradient_penalty = calc_gradient_penalty(self.D, self.input, fake_img, self.bs)
                d_err = d_fake_err + d_real_err + gradient_penalty

                d_err.backward()
                self.optim_d.module.step()
                """
               
                ####################
                #update G
                ###################
                self.freezeD()

                #fake_img, fake_noise = self.G(self.input)
                fake_img, fake_noise = self.G(self.Input_with_noise)
                rec_loss = self.mse_loss(fake_img, self.input) * self.w_con
                g_adv = self.l_adv(self.D(fake_img)[0], self.g_real_label) * self.w_adv

                gen_err = rec_loss + g_adv
                gen_err.backward()
                self.optim_g.module.step()

                self.add_loss(gen_err, "gen_err")

                """
                这部分是有梯度惩罚项的
                gen_err = -torch.mean(self.D(fake_img))

                d_fake_err = torch.mean(self.D(fake_img.detach()))
                d_real_err = -torch.mean(self.D(self.input))
                d_fake_err = torch.mean(self.D(fake_img.detach()))
                d_err = d_fake_err + d_real_err

                gen_err.backward()
                self.optim_g.module.step()
                """
                
                print(f"[{epoch}|{self.epoches}] [{i}|{len(self.train_data)}]d_err = {d_err.item()}, d_fake_err = {d_fake_err.item()}, d_real_err={d_real_err.item()} g_err = {gen_err.item()}, z_err = {z_err.item()}")
                #print(f"Dis: [{epoch}|{self.epoches}] [{i}|{len(self.train_data)}] | d_err = {d_err.item()} , g_err = {gen_err.item()} | d_fake_err = {d_fake_err.item()}, d_real_err={d_real_err.item()} z_err = {z_err.item()}")
                
                #del fake_img
                # del sigma
                #del self.input
                # del self.Input_with_noise
                #del z_err
                del d_real_err
                del d_fake_err
                # del gradient_penalty
                del d_err
               
            """
            #####################
            #update G
            ####################
            self.freezeD()
            for i, data in enumerate(self.train_data):
                self.input, _ = data
                if torch.cuda.is_available():
                    self.input = self.input.cuda(non_blocking=True)
                
                self.optim_g.zero_grad()
                # fake_img, _ = self.G(self.Input_with_noise)
                fake_img, fake_noise = self.G(self.input)
                
                z_err = 0.5 *(self.l_lat(self.DL(self.real_z_data), self.real_z_label) + self.l_lat(self.DL(torch.squeeze(fake_noise.detach())),self.fake_z_label))
                
                # 这部分是用传统的来做
                pred_fake = self.D(fake_img)
                pred_real = self.D(self.input)

                d_fake_err = self.l_adv(torch.squeeze(self.D(fake_img)), self.d_fake_label)
                d_real_err = self.l_adv(torch.squeeze(self.D(self.input)), self.d_real_label)
                d_err = 0.5 * (d_fake_err + d_real_err)

                rec_loss = self.mse_loss(fake_img, self.input) * self.w_con
                #print(f"d_fake_err = {d_fake_err.shape}, self.g_real_label = {self.g_real_label.shape}")
                #g_adv = self.l_adv(torch.squeeze(self.D(fake_img)), self.g_real_label) * self.w_adv
                g_adv = self.l_adv(self.D(fake_img), self.g_real_label) * self.w_adv

                gen_err = rec_loss + g_adv

                
                # 这部分是有梯度惩罚项的
                # gen_err = -torch.mean(self.D(fake_img))

                # d_fake_err = torch.mean(self.D(fake_img.detach()))
                # d_real_err = -torch.mean(self.D(self.input))
                # d_fake_err = torch.mean(self.D(fake_img.detach()))
                # d_err = d_fake_err + d_real_err

                # gen_err.backward()
                # self.optim_g.module.step()
                
                gen_err.backward()
                self.optim_g.module.step()
                
                print(f"[{epoch}|{self.epoches}] [{i}|{len(self.train_data)}]d_err = {d_err.item()}, d_fake_err = {d_fake_err.item()}, d_real_err={d_real_err.item()} g_err = {gen_err.item()}, z_err = {z_err.item()}")
                #self.add_loss()

                #del d_err
                #del d_fake_err
                #del d_real_err
                #del gen_err
            """

            # 保存图片
            os.makedirs(os.path.join(self.trainImg_dir, "real"), exist_ok=True)
            os.makedirs(os.path.join(self.trainImg_dir, "fake"), exist_ok=True)
            os.makedirs(os.path.join(self.trainImg_dir, "noise"), exist_ok=True)
            # os.makedirs(os.path.join(self.trainImg_dir, "noise"), exist_ok=True)
            if epoch % self.print_freq == 0:
               vutils.save_image(self.input[0:self.image_grids_numbers], self.trainImg_dir + "/real/epoch_{}.png".format(epoch), nrow=self.n_row_in_grid, normalize=True )
               vutils.save_image(fake_img[0:self.image_grids_numbers], self.trainImg_dir + "/fake/epoch_{}.png".format(epoch),nrow=self.n_row_in_grid, normalize=True)
               vutils.save_image(self.Input_with_noise[0:self.image_grids_numbers], self.trainImg_dir + "/noise/epoch_{}.png".format(epoch),nrow=self.n_row_in_grid, normalize=True)
            #    vutils.save_image(self.Input_with_noise[0:self.image_grids_numbers], self.trainImg_dir + "/noise/epoch_{}.png".format(epoch),nrow=self.n_row_in_grid, normalize=True)
            
            if epoch % 20 == 0 or epoch >= self.epoches -3:
                self.save_weights(epoch)
        self.dict2json()
        
    def dict2json(self):
        with open(os.path.join(self.loss_dir, self.opt.name + "_loss.json"), "w") as f:
            json.dump(self.errors, f)
        
    def add_loss(self, err, name):
        if name not in self.errors.keys():
            self.errors[name] = list()
        self.errors[name].append(err.item())

    def save_weights(self, epoch: int, is_best: bool = False):
        torch.save({'epoch': epoch, 'state_dict': self.D.module.state_dict()},
                    f"{self.weight_dir}/{self.opt.name}_netD_{epoch}.pth")
        torch.save({'epoch': epoch, 'state_dict': self.G.module.state_dict()},
                    f"{self.weight_dir}/{self.opt.name}_netG_{epoch}.pth")
        ##
        torch.save({'epoch': epoch, 'state_dict': self.DL.module.state_dict()},
                    f"{self.weight_dir}/{self.opt.name}_netDL_{epoch}.pth")

    def load_weights(self, epoch=None, is_best: bool = False, path=None):

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        path_g = f"{self.weight_dir}/{self.opt.name}_netG_{epoch}.pth"
        path_d = f"{self.weight_dir}/{self.opt.name}_netD_{epoch}.pth"
        path_dl = f"{self.weight_dir}/{self.opt.name}_netDL_{epoch}.pth"
            #
        #print(f"path_g={path_g}")
        print('>> Loading weights...')
        # print(f"path_g = f{path_g}")
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        ##
        weights_dl = torch.load(path_dl)['state_dict']
  
        from collections import OrderedDict
        new_weight_g = OrderedDict()
        for k, v in weights_g.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_weight_g[k] = v

        try:
            self.netg = nn.DataParallel(self.G.cuda())
            #self.netg = self.netg.module
            #self.netg.load_state_dict(weights_g)
            self.netg.module.load_state_dict(new_weight_g)
            # print(f"weights_g ={weights_d}")
            #self.netd = nn.DataParallel(self.D.cuda())
            #self.netd.load_state_dict(weights_d)

            #self.netDL = nn.DataParallel(self.DL.cuda())
            #self.netDL.load_state_dict(weights_dl)
        except IOError:
            raise IOError("netG weights not found")
        print('Done.')


    def Test(self):
        self.an_scores = list()
        #real_label = list()

        realDir = os.path.join(self.test_dir, "real")
        fakeDir = os.path.join(self.test_dir, "fake")
        os.makedirs(realDir, exist_ok=True)
        os.makedirs(fakeDir, exist_ok=True)

        # 下载权重
        self.load_weights(epoch=self.epoches-1)        
        idx = 0
        #print(f"self.test_data.shape={self.test_data.shape}")
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_data):
                #print(f"real_data.shape={data.shape}")
                #data, _ = real_data
                if torch.cuda.is_available():
                    data = data.cuda(non_blocking=True)
                
                genImg,_ = self.G(data)
                N, C, H, W = data.shape
                res = (data - genImg).view(N, -1)
                res = torch.mean(torch.pow(res, 2),dim=1)
                #print(f"res.shape={res.shape}")
                #print(f"self/real_label.shape ={self.real_label.shape}")
                for r in res:
                    self.an_scores.append(r)
                #print(f"res = {self.an_scores}")
                #self.real_label.append(label)

                for i in range(len(genImg)):
                    vutils.save_image(data[i], realDir +"12_" + str(i) + ".png", normalize=True)
                    vutils.save_image(genImg[i], fakeDir +"_"+ str(i) + ".png", normalize=True)

            self.an_scores = torch.tensor(self.an_scores)
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc_plot(label=self.real_label, scores=self.an_scores, name=self.name, savePath=os.path.join(self.test_dir, "auc"), save=True)
            print(f"{self.name} AUC is :{auc}")






                

            

