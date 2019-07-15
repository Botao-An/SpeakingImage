import torch

class Val:


    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def run_epoch(self, iteration_loss=False):

        self.model.eval()
        epoch_loss = 0.0
        for step, batch_data in enumerate(self.data_loader):
            cover = batch_data[0].to(self.device)
            secret = batch_data[1].to(self.device)

            with torch.no_grad():

                stego_image, restored_secret = self.model(cover, secret)

                loss1 = self.criterion(stego_image, cover)
                loss2 = self.criterion(restored_secret, secret)
                loss = loss1 + loss2

            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration total testing loss: %.6f; Image loss: %.6f; Audio loss: %.6f " % (step, loss.item(), loss1.item(), loss2.item()))

        return epoch_loss/len(self.data_loader)