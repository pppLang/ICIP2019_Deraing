import torch
from utilities import batch_RMSE_G, batch_PSNR, get_ssim


def train_epoch(model,
                optimizer,
                train_loader,
                l1_criterion,
                mask_criterion,
                ssim_criterion,
                epoch,
                writer=None,
                radio=1):
    model.train()
    num = len(train_loader)
    for i, data in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        O, B, M = data['O'].cuda(), data['B'].cuda(), data['M'].cuda()
        recon = model.forward(O)
        # recon = model.forward(recon)
        recon = torch.clamp(recon, 0., 1.)
        l1_loss = l1_criterion(recon, B)
        # mask_loss = mask_criterion(mask[:, 0, :, :], M)
        ssim_loss = ssim_criterion(recon, B)
        loss = l1_loss - radio * ssim_loss  # + mask_loss
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            with torch.no_grad():
                rmse = batch_RMSE_G(B, recon)
                psnr = batch_PSNR(B, recon)
                print('epoch {}, [{}/{}], loss {}, PSNR {}, SSIM {}, RMSE {}, '.format(epoch, i, num, loss, psnr, ssim_loss.item(), rmse))
                if writer is not None:
                    step = epoch * num + i
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('l1_loss', l1_loss.item(), step)
                    writer.add_scalar('ssim', ssim_loss.item(), step)
                    # writer.add_scalar('mask_loss', mask_loss.item(), step)
                    writer.add_scalar('RMSE', rmse.item(), step)
                    writer.add_scalar('PSNR', psnr.item(), step)


def test(model, test_dataset, test_file_num, epoch, writer=None):
    model.eval()
    if test_file_num is None:
        test_file_num = len(test_dataset)
    rmse_sum, psnr_sum, ssim_sum = 0, 0, 0
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
            # print(O.shape, B.shape, M.shape, mask.shape, recon.shape)
            print('epoch {}, test img {}, rmse {}, psnr {}, ssim {}'.format(epoch, i, rmse, psnr, ssim))
            rmse_sum += rmse
            psnr_sum += psnr
            ssim_sum += ssim

    print('\nepoch {}, avg RMSE {}, avg PSNR {}, avg SSIM {}\n'.format(
        epoch, rmse_sum / test_file_num, psnr_sum / test_file_num, ssim_sum / test_file_num))
    if writer is not None:
        writer.add_scalar('test_RMSE', rmse_sum / test_file_num, epoch)
        writer.add_scalar('test_PSNR', psnr_sum / test_file_num, epoch)
        writer.add_scalar('test_SSIM', ssim_sum / test_file_num, epoch)
