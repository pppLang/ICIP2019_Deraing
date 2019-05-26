import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as udata
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import TrainDataset, TestDataset2
from generate_result import generate_result
# from SPANet import SPANet_v2
# from PRENet import PReNet_v2
# from UNet import UNet_v2
from UNet.UNet_v2 import UNet_v2
import time
from train import train_epoch, test
from utilities import SSIM

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--batchSize_per_gpu", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=5e-3, help="initial learning rate")
parser.add_argument("--gpus", type=str, default="5,6,7", help='path log files')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

device_ids = list(range(len(opt.gpus.split(','))))
print('device_ids {}'.format(device_ids))
batch_size = opt.batchSize_per_gpu * len(device_ids)
interval = 20
weight_decay = 1e-5
radio = 1

outf = 'logs2/UNetv2_{}_{}_{}_{}_{}_{}'.format(opt.lr, weight_decay, radio, opt.batchSize_per_gpu, len(device_ids), interval)

print(outf)
print(os.environ["CUDA_VISIBLE_DEVICES"])


def main():
    print(outf)
    print("loading dataset ...")
    trainDataset = TrainDataset(name='/data0/niejiangtao/ICIP2019Deraining/train/train.h5')
    batchSize = opt.batchSize_per_gpu * len(device_ids)
    trainLoader = udata.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)

    testDataset = TestDataset2(name='/data0/niejiangtao/ICIP2019Deraining/test_a/test.h5')
    print('testDataset len : {}'.format(len(testDataset)))
    
    l1_criterion = nn.L1Loss().cuda()
    # mask_criterion = nn.MSELoss().cuda()
    ssim_criterion = SSIM().cuda()
    
    model = UNet_v2(n_channels=3, n_classes=3)
    # model = RESCAN()

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    
    beta1 = 0.9
    beta2 = 0.999
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=weight_decay, betas=(beta1, beta2))

    writer = SummaryWriter(outf)

    for epoch in range(opt.epochs):
        start = time.time()
        current_lr = opt.lr / 2**int(epoch / interval)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("epoch {} learning rate {}".format(epoch, current_lr))

        # test(model, testDataset, None, epoch, writer=writer)
        train_epoch(model, optimizer, trainLoader, l1_criterion, None, ssim_criterion, epoch, writer=writer, radio=radio)

        if (epoch+1) % 5 == 0:
            test(model, testDataset, None, epoch, writer=writer)
        if (epoch+1) % 20 == 0:
            """ torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(outf, 'checkpoint_{}.pth'.format(epoch))) """
            torch.save(model.state_dict(), os.path.join(outf, 'model_{}.pth'.format(epoch)))

        end = time.time()
        print('epoch {} cost {} hour '.format(
            epoch, str((end - start) / (60 * 60))))
            
    torch.save(model.state_dict(), os.path.join(outf, 'model.pth'))
    generate_result(model, outf, testDataset, mat=False, ouput_img=True)


if __name__ == "__main__":
    main()
