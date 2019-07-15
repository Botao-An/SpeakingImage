# import visdom
# vis = visdom.Visdom(env='step train loss')

class Train:

    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self, iteration_loss=False):

        self.model.train()
        epoch_loss = 0.0

        for step, batch_data in enumerate(self.data_loader):
            cover = batch_data[0].to(self.device)
            secret = batch_data[1].to(self.device)

            stego_image, resotred_secret = self.model(cover, secret)

            loss1 = self.criterion(stego_image, cover)
            loss2 = self.criterion(resotred_secret, secret)
            loss = loss1+loss2

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration total training loss: %.6f; Image loss: %.6f; Audio loss: %.6f " % (step, loss.item(), loss1.item(), loss2.item()))

            # if (step + 1) % 10 == 0:
            #     vis.line(X=torch.FloatTensor([step]), Y=torch.FloatTensor([loss]), win='step train',
            #          update='append' if step > 1 else None, opts={'title': 'step train loss'})


        return epoch_loss/len(self.data_loader)

