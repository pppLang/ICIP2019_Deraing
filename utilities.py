import os
import torch
import torch.nn as nn
import numpy as np
# from GETPSNR_SAM import PSNR_GPU
import gc

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F


def get_testfile_list():
    path = '/data0/langzhiqiang/data'  #'D:\\'#
    test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')

    test_rgb_filename_list = []
    for line in test_names_file:
        line = line.split('/n')[0]
        hyper_rgb = line.split(' ')[0]
        test_rgb_filename_list.append(hyper_rgb)
        # print(hyper_rgb)

    return test_rgb_filename_list


def Loss_SAM(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    nom = torch.sum(torch.mul(im_true, im_fake), dim=1)
    denom1 = torch.sqrt(torch.sum(torch.pow(im_true, 2), dim=1))
    denom2 = torch.sqrt(torch.sum(torch.pow(im_fake, 2), dim=1))
    sam = torch.acos(torch.div(nom, torch.mul(denom1, denom2)))
    sam = torch.mul(torch.div(sam, np.pi), 180)
    sam = torch.div(torch.sum(sam), N * H * W)
    return sam


def batch_PSNR(im_true, im_fake, data_range=255):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
    psnr = 10. * torch.log((data_range**2) / err) / np.log(10.)
    return torch.mean(psnr)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def batch_SAM(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C, H * W)
    Ifake = im_fake.clone().resize_(N, C, H * W)
    nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H * W)
    denom1 = torch.pow(Itrue, 2).sum(dim=1).sqrt_().resize_(N, H * W)
    denom2 = torch.pow(Ifake, 2).sum(dim=1).sqrt_().resize_(N, H * W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H * W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (N * H * W)
    return sam


def batch_RMSE(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sqrt_().mul_(data_range).sum(
        dim=1, keepdim=True).div_(C * H * W)
    return torch.mean(err)


def batch_RMSE_G(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W).sqrt_()
    return torch.mean(err)


def batch_rRMSE(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sqrt_().div_(Itrue).sum(
        dim=1, keepdim=True).div_(C * H * W)
    return torch.mean(err)


def batch_rRMSE_G(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    gt = torch.mean(Itrue).pow(2)
    err = mse(Itrue, Ifake).div_(gt).sum(
        dim=1, keepdim=True).div_(C * H * W).sqrt_()
    return torch.mean(err)


def plot_spectrum(real, fake):
    x = np.linspace(1, 31, 31, endpoint=True)
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    plot_real, = ax.plot(x, real)
    plot_fake, = ax.plot(x, fake)
    fig.legend((plot_real, plot_fake), ('real', 'fake'))
    canvas.draw()
    I = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
    I = I.reshape(canvas.get_width_height()[::-1] + (3, ))
    I = np.transpose(I, [2, 0, 1])
    return np.float32(I)


def weights_init_kaimingUniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.uniform(m.weight.data, a=0, b=1)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.uniform(m.weight.data, a=0, b=1)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_kaimingNormal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
        pass
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_xavierNormal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)


def validate(model, writer, mode, testDataset, step, epoch, i, trainLoader):
    model.eval()
    print("computing results on validation {} set ...".format(mode))
    num = len(testDataset)
    # average_RMSE = 0.
    average_RMSE_G = 0.
    # average_rRMSE = 0.
    average_rRMSE_G = 0.
    average_SAM = 0.
    average_PSRN = 0.
    for k in range(num):
        # data
        real_hyper, _, real_rgb = testDataset[k]
        """ print('test real rgb shape : {}'.format(real_rgb.shape))
        print(real_hyper.shape) """
        real_hyper.unsqueeze_(0)
        real_rgb.unsqueeze_(0)
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()
        # forward
        with torch.no_grad():
            fake_hyper = model.forward(real_rgb)
        # metrics
        # RMSE = batch_RMSE(real_hyper, fake_hyper)
        RMSE_G = batch_RMSE_G(real_hyper, fake_hyper)
        # rRMSE = batch_rRMSE(real_hyper, fake_hyper)
        rRMSE_G = batch_rRMSE_G(real_hyper, fake_hyper)
        SAM = batch_SAM(real_hyper, fake_hyper)
        PSNR = batch_PSNR(real_hyper, fake_hyper)
        # average_RMSE    += RMSE.item()
        average_RMSE_G += RMSE_G.item()
        # average_rRMSE   += rRMSE.item()
        average_rRMSE_G += rRMSE_G.item()
        average_SAM += SAM.item()
        average_PSRN += PSNR.item()

    del real_hyper, real_rgb, fake_hyper
    gc.collect()

    # writer.add_scalar('RMSE_{}'.format(mode), average_RMSE/num, step)
    writer.add_scalar('RMSE_{}'.format(mode), average_RMSE_G / num, step)
    # writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE/num, step)
    writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE_G / num, step)
    writer.add_scalar('SAM_{}'.format(mode), average_SAM / num, step)
    writer.add_scalar('PSNR_{}'.format(mode), average_PSRN / num, step)
    print(
        "[epoch %d][%d/%d] validation: RMSE_G: %.4f PSNR: %.4f rRMSE_G: %.4f SAM: %.4f"
        % (epoch, i, len(trainLoader), average_RMSE_G / num,
           average_PSRN / num, average_rRMSE_G / num, average_SAM / num))


def validate_CAVE(model, writer, mode, testDataset, step, epoch, i,
                  trainLoader):
    model.eval()
    print("computing results on validation {} set ...".format(mode))
    num = len(testDataset)
    # average_RMSE = 0.
    average_RMSE_G = 0.
    # average_rRMSE = 0.
    average_rRMSE_G = 0.
    average_SAM = 0.
    average_PSRN = 0.

    path = '/data0/langzhiqiang/CAVE_RGB/'
    test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')
    # num = len(test_names_file)
    print(len(testDataset))
    model.eval()
    # print(len(test_names_file))
    for j, line in enumerate(test_names_file):
        line = line.split('\n')[0]
        key = line.split('/')[-1].split('.')[0]
        real_hyper, _, real_rgb = testDataset[key]
        """ print('test real rgb shape : {}'.format(real_rgb.shape))
        print(real_hyper.shape) """
        real_hyper.unsqueeze_(0)
        real_rgb.unsqueeze_(0)
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()
        # forward
        with torch.no_grad():
            fake_hyper = model.forward(real_rgb)
        if isinstance(fake_hyper, tuple):
            fake_hyper = fake_hyper[0]
        # metrics
        # RMSE = batch_RMSE(real_hyper, fake_hyper)
        # RMSE_G = batch_RMSE_G(real_hyper, fake_hyper)
        # rRMSE = batch_rRMSE(real_hyper, fake_hyper)
        # rRMSE_G = batch_rRMSE_G(real_hyper, fake_hyper)
        # SAM = batch_SAM(real_hyper, fake_hyper)
        PSNR = batch_PSNR(real_hyper, fake_hyper)
        # average_RMSE    += RMSE.item()
        # average_RMSE_G  += RMSE_G.item()
        # average_rRMSE   += rRMSE.item()
        # average_rRMSE_G += rRMSE_G.item()
        # average_SAM     += SAM.item()
        average_PSRN += PSNR.item()
        print('img {}, psnr {}'.format(j, PSNR))

    del real_hyper, real_rgb, fake_hyper
    gc.collect()

    # writer.add_scalar('RMSE_{}'.format(mode), average_RMSE/num, step)
    # writer.add_scalar('RMSE_{}'.format(mode), average_RMSE_G/num, step)
    # writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE/num, step)
    # writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE_G/num, step)
    # writer.add_scalar('SAM_{}'.format(mode), average_SAM/num, step)
    writer.add_scalar('PSNR_{}'.format(mode), average_PSRN / num, step)
    print(
        "[epoch %d][%d/%d] validation: RMSE_G: %.4f PSNR: %.4f rRMSE_G: %.4f SAM: %.4f"
        % (epoch, i, len(trainLoader), average_RMSE_G / num,
           average_PSRN / num, average_rRMSE_G / num, average_SAM / num))


def validate_Harvard(model, writer, mode, testDataset, step, epoch, i,
                     trainLoader):
    model.eval()
    print("computing results on validation {} set ...".format(mode))
    num = len(testDataset)
    # average_RMSE = 0.
    average_RMSE_G = 0.
    # average_rRMSE = 0.
    average_rRMSE_G = 0.
    average_SAM = 0.
    average_PSRN = 0.

    path = '/data0/langzhiqiang/Harvard_RGB/'
    test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')
    # num = len(test_names_file)
    print(len(testDataset))
    model.eval()
    # print(len(test_names_file))
    for j, line in enumerate(test_names_file):
        line = line.split('\n')[0]
        key = line.split('/')[-1].split('.')[0]
        real_hyper, _, real_rgb = testDataset[key]
        """ print('test real rgb shape : {}'.format(real_rgb.shape))
        print(real_hyper.shape) """
        real_hyper.unsqueeze_(0)
        real_rgb.unsqueeze_(0)
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()
        # forward
        with torch.no_grad():
            fake_hyper = model.forward(real_rgb)
        if isinstance(fake_hyper, tuple):
            fake_hyper = fake_hyper[0]
        # metrics
        # RMSE = batch_RMSE(real_hyper, fake_hyper)
        # RMSE_G = batch_RMSE_G(real_hyper, fake_hyper)
        # rRMSE = batch_rRMSE(real_hyper, fake_hyper)
        # rRMSE_G = batch_rRMSE_G(real_hyper, fake_hyper)
        # SAM = batch_SAM(real_hyper, fake_hyper)
        PSNR = batch_PSNR(real_hyper, fake_hyper)
        # average_RMSE    += RMSE.item()
        # average_RMSE_G  += RMSE_G.item()
        # average_rRMSE   += rRMSE.item()
        # average_rRMSE_G += rRMSE_G.item()
        # average_SAM     += SAM.item()
        average_PSRN += PSNR.item()
        print('img {}, psnr {}'.format(j, PSNR))

    del real_hyper, real_rgb, fake_hyper
    gc.collect()

    # writer.add_scalar('RMSE_{}'.format(mode), average_RMSE/num, step)
    # writer.add_scalar('RMSE_{}'.format(mode), average_RMSE_G/num, step)
    # writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE/num, step)
    # writer.add_scalar('rRMSE_{}'.format(mode), average_rRMSE_G/num, step)
    # writer.add_scalar('SAM_{}'.format(mode), average_SAM/num, step)
    writer.add_scalar('PSNR_{}'.format(mode), average_PSRN / num, step)
    print(
        "[epoch %d][%d/%d] validation: RMSE_G: %.4f PSNR: %.4f rRMSE_G: %.4f SAM: %.4f"
        % (epoch, i, len(trainLoader), average_RMSE_G / num,
           average_PSRN / num, average_rRMSE_G / num, average_SAM / num))


class WeisLoss(nn.Module):
    def __init__(self):
        super(WeisLoss, self).__init__()

    def forward(self, weis_out):
        weis_out = torch.squeeze(weis_out)
        weis_loss = torch.div(
            torch.ones_like(weis_out) * 1e-8, (weis_out - 0.5).pow(2)).mean()
        return weis_loss