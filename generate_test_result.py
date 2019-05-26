import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import cv2
import numpy as np
from utilities import batch_RMSE_G, batch_PSNR, get_ssim
from UNet.UNet_v11 import UNet_v11
from dataset import TestDataset2

""" os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4" """


def generate_result(model, outf, test_file_pathes):
    model.eval()
    test_file_num = len(test_file_pathes)
    if not os.path.exists(os.path.join(outf, 'derained_results')):
        os.makedirs(os.path.join(outf, 'derained_results'))
    if not os.path.exists(os.path.join(outf, 'test_input')):
        os.makedirs(os.path.join(outf, 'test_input'))
        
    with torch.no_grad():
        for i in range(test_file_num):
            test_file_path = test_file_pathes[i]
            img_name = test_file_path.split('/')[-1]
            rained = cv2.imread(test_file_path)
            print(rained.max(), rained.min())
            rained = np.transpose(rained.astype(np.float32) / 255, [2, 0, 1])
            print(rained.max(), rained.min())
            
            rained = torch.Tensor(rained).unsqueeze(0) #  .cuda()
            recon = model.forward(rained)
            recon = recon[0, :, :, :].cpu().numpy()
            recon = np.transpose(recon, [1, 2, 0])
            recon = recon * 255
            print(recon.max(), recon.min())
            print('img {}, {}, shape {}'.format(i, img_name, recon.shape))
            cv2.imwrite(os.path.join(outf, 'derained_results', img_name), recon)


if __name__ == "__main__":
    """ device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')
    device3 = torch.device('cuda:2')
    device4 = torch.device('cuda:3')
    device5 = torch.device('cuda:4') """
    outf = '/home/langzhiqiang/MyDerainer/logs/UNetv11_0.005_32_3_20'
    # model = UNet_v11_test(3, 3, device1, device2, device3, device5, device4)
    model = UNet_v11(3, 3)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(outf, 'model.pth'), map_location=torch.device('cuda:0')))
    model = model.module
    # model.cuda()
    # model.move()
    model.cpu()

    test_root_path = '/data0/niejiangtao/ICIP2019Deraining/Test_data'
    # test_root_path = '/data0/niejiangtao/ICIP2019Deraining/test_a/data'
    import glob
    test_file_pathes = glob.glob(os.path.join(test_root_path, '*.jpg'))
    import random
    random.shuffle(test_file_pathes)

    generate_result(model, outf, test_file_pathes)