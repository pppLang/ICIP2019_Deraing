import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import batch_RMSE_G, batch_PSNR, get_ssim
from UNet.UNet_v11 import UNet_v11
from dataset import TestDataset2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_result(model, outf, test_dataset, mat=False, ouput_img=True):
    model.eval()
    test_file_num = len(test_dataset)
    if ouput_img and not os.path.exists(os.path.join(outf, 'output_imgs')):
        os.makedirs(os.path.join(outf, 'output_imgs'))
    results = []
    with torch.no_grad():
        for i in range(test_file_num):
            data = test_dataset[i]
            O, B, M = data['O'], data['B'], data['M']
            O = torch.Tensor(O).unsqueeze(0).cuda()
            B = torch.Tensor(B).unsqueeze(0).cuda()
            M = torch.Tensor(M).unsqueeze(0).cuda()
            recon = model.forward(O)
            # recon = model.forward(recon)
            recon = torch.clamp(recon, 0., 1.)
            rmse = batch_RMSE_G(B, recon)
            psnr = batch_PSNR(B, recon)
            ssim = get_ssim(B, recon)
            print('test img {}, rmse {}, psnr {}, ssim {}'.format(i, rmse, psnr, ssim))
            results.append([rmse, psnr, ssim])
            if ouput_img:
                generate_imgs(O, B, recon, outf, i, psnr)

    results = np.array(results)
    result_avg = np.sum(results, axis=0) / test_file_num
    print(results.shape, result_avg.shape)
    print('average results : {}'.format(result_avg))
    results = np.append(results, [result_avg], axis=0)

    excel = pd.DataFrame(data=results, columns=['rmse', 'psnr', 'ssim'])
    excel.to_csv(os.path.join(outf, 'results.csv'))


def generate_imgs(O, B, recon, outf, img_id, psnr):
    O = O[0, :, :, :].cpu().numpy()
    O = np.transpose(O, [1, 2, 0])
    B = B[0, :, :, :].cpu().numpy()
    B = np.transpose(B, [1, 2, 0])
    recon = recon[0, :, :, :].cpu().numpy()
    recon = np.transpose(recon, [1, 2, 0])
    mse = np.sqrt(np.mean(np.square((recon - B)), axis=2, keepdims=False))
    
    plt.imshow(O)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('img{}_input'.format(img_id))
    plt.savefig(os.path.join(outf, 'output_imgs', 'img_{}_input.png'.format(img_id)))
    plt.close()

    plt.imshow(B)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('img{}_GT'.format(img_id))
    plt.savefig(os.path.join(outf, 'output_imgs', 'img_{}_GT.png'.format(img_id)))
    plt.close()
    
    plt.imshow(recon)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('img{}_{}'.format(img_id, str(psnr.item())))
    plt.savefig(os.path.join(outf, 'output_imgs', 'img_{}_recon.png'.format(img_id)))
    plt.close()

    plt.imshow(mse)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.title('img{}_mse'.format(img_id))
    plt.savefig(os.path.join(outf, 'output_imgs', 'img_{}_mse.png'.format(img_id)))
    plt.close()


if __name__ == "__main__":
    outf = '/home/langzhiqiang/MyDerainer/logs/UNetv2Recurrent_0.005_32_3_20'
    model = UNet_v11(n_channels=3, n_classes=3)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(outf, 'model.pth'), map_location=torch.device('cuda:0')))
    model = model.module
    model.cuda()

    testDataset = TestDataset2(name='/data0/niejiangtao/ICIP2019Deraining/test_a/test.h5')
    print('testDataset len : {}'.format(len(testDataset)))

    generate_result(model, outf, testDataset, mat=False, ouput_img=True)