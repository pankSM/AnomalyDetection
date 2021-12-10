""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
   [argparse]: Class containing argparse
"""

import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist | anomaly ')
        self.parser.add_argument('--train_root', default='', help='path to dataset')
        self.parser.add_argument('--test_root', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--test_batchsize', type=int, default=2, help='input batch size')

        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--lat_dim', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--loss_type', type=str, default='l1', help='folder l1|mse|ssim|mssim')
        self.parser.add_argument('--optim_type', type=str, default='Adam', help='folder Adam|SGD|RMSprop')
        self.parser.add_argument('--norm_type', type=str, default='batch', help='folder |batch|instance')
        self.parser.add_argument('--init_type', type=str, default='norm', help='net inittialize |norm|')

        self.parser.add_argument('--niter', type=int, default=90, help='the epoch used to test')
        self.parser.add_argument('--netsize', type=int, default=64, help='the size of the input image.64 | 128 | 256 ')

        self.parser.add_argument('--sigma_noise', type=float, default=0.9, help="the noise add to image")
        self.parser.add_argument('--image_grids_numbers', default=2, type=int,
                            help='total number of grid squares to be saved every / few epochs')
        self.parser.add_argument('--n_row_in_grid', default=1, type=int,
                            help=' Number of images displayed in each row of the grid images.')

        self.parser.add_argument('--g_dim', type=int, default=64)
        self.parser.add_argument('--d_dim', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--seed_value', type=int, default=9999, help='randdom seed')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='randdom seed')


        self.parser.add_argument('--name', type=str, default='object name', help='name of object')
        self.parser.add_argument('--model', type=str, default='skipganomaly',
                                 help='chooses which model to use. ganomaly')

        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')

        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--test_block_path', default='./output', help='folder to output images and model checkpoints')

        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100,
                                 help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')

        # 生成器和判别器使用不同的学习率
        self.parser.add_argument('--d_lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--g_lr', type=float, default=0.0003, help='initial learning rate for adam')
        self.parser.add_argument('--d_lat_lr', type=float, default=0.0001, help='initial learning rate for adam')

        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        return self.opt
