import torch
import scipy.io as io
import os
import numpy as np

class Test:

    def __init__(self, model, data_loader, criterion, device, outimage_path, outaudio_path):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.outimage_path = outimage_path
        self.outaudio_path = outaudio_path

    def run_epoch(self, iteration_loss=False):

        self.model.eval()
        epoch_loss = 0.0
        n = 0

        for step, batch_data in enumerate(self.data_loader):

            n = n+1

            cover = batch_data[0].to(self.device)
            secret = batch_data[1].to(self.device)

            cover_path =  os.path.join(self.outimage_path, 'cover/')
            secret_path = os.path.join(self.outaudio_path, 'secret/')


            stego_image_path = os.path.join(self.outimage_path, 'stego_image/')
            restored_secret_path = os.path.join(self.outaudio_path, 'restored_secret/')

            io.savemat(cover_path+str(n)+'.mat', {'cover':np.array(cover.cpu())})
            io.savemat(secret_path + str(n) + '.mat', {'secret': np.array(secret.cpu())})

            with torch.no_grad():

                stego_image, restored_secret = self.model(cover, secret)

                io.savemat(stego_image_path + str(n) + '.mat', {'stego_image': np.array(stego_image.cpu())})
                io.savemat(restored_secret_path + str(n) + '.mat', {'restored_secret': np.array(restored_secret.cpu())})

                loss1 = self.criterion(stego_image, cover)
                loss2 = self.criterion(restored_secret, secret)
                loss = loss1 + loss2

            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration total testing loss: %.6f; Image loss: %.6f; Audio loss: %.6f " % (step, loss.item(), loss1.item(), loss2.item()))

        return epoch_loss/len(self.data_loader)