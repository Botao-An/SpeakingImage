import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from train import Train
from test import Test
from val import Val
from Model.dazhuang import DaZhuangNet
from args import get_arguments
import utils


# import visdom
# vis = visdom.Visdom(env='epoch loss')

args = get_arguments()

device = torch.device(args.device)

def load_dataset(dataset):
    print("\n Loading dataset... \n")

    print("Select data directory:", args.data_dir)
    print("Save directory:", args.save_dir)

    cover_mu = [0.5, 0.5, 0.5]
    cover_sigma = [0.5, 0.5, 0.5]
    #
    # secret_mu = [0.5]
    # secret_sigma = [0.5]

    cover_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((300,300)),
        transforms.RandomCrop((256,256), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(cover_mu, cover_sigma)
    ])

    # secret_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(secret_mu, secret_sigma)
    # ])

    train_set = dataset(
        args.data_dir,
        cover_transform=cover_transform,
        secret_transform=None
    )

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_set = dataset(
        args.data_dir,
        mode='val',
        cover_transform=cover_transform,
        secret_transform=None
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_set = dataset(
        args.data_dir,
        mode='test',
        cover_transform=cover_transform,
        secret_transform=None
    )

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Train dataset size: ", len(train_set))
    print("Validation dataset size:", len(val_set))
    print("Test dataset size:", len(test_set))

    if args.mode.lower() == 'test':
        cover, secret = iter(test_loader).next()

    else:
        cover, secret = iter(train_loader).next()

    print("Image size:", cover.size())
    print("Audio size:", secret.size())

    return train_loader, val_loader, test_loader

def train(train_loader, val_loader):

    print("\nTraining...\n")

    model = DaZhuangNet().to(device)

    print(model)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr = args.learning_rate,
        weight_decay=args.weight_decay
    )

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    start_epoch = 0
    min_loss = float('inf')

    print()
    train = Train(model, train_loader, optimizer, criterion, device)
    val = Val(model, val_loader, criterion, device)

    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()

        epoch_loss = train.run_epoch(iteration_loss=True)
        print(">>>> [Epoch: {0:d}] Avg. loss on training data: {1: .6f}".format(epoch, epoch_loss))

        # if (epoch+1)%5 == 0 or epoch+1==args.epochs:
        print(">>>> [Epoch: {0:d}] Validation".format(epoch))
        loss = val.run_epoch(iteration_loss=True)
        print(">>>> [Epoch: {0:d}] Avg. loss on validation data: {1: .6f}".format(epoch, loss))

        # vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss]), win='epoch train',
        #          update='append' if epoch > 1 else None, opts={'title': 'epoch validation loss'})

        if loss<min_loss:
            print("\nWaiting for the best model... Saving the current model...\n")
            min_loss=loss
            utils.save_checkpoint(model, optimizer, epoch+1, min_loss, args)

    return model

def test(model, test_loader):
    print("\nTesting...\n")

    critertion = nn.MSELoss()

    outimage_path = os.path.join(args.data_dir, 'outimage')
    outaudio_path = os.path.join(args.data_dir, 'outaudio')

    test = Test(model, test_loader, critertion, device, outimage_path, outaudio_path)

    print(">>>>Running test dataset")

    loss = test.run_epoch(iteration_loss=True)

    print(">>>>Avg. loss on test data: {0:.6f}".format(loss))



if __name__ == '__main__':

    assert os.path.isdir(args.data_dir), "The directory \"{0}\" doesn't exist.".format(args.data_dir)

    from Data.mydataset import MyDataset as dataset

    train_loader, val_loader, test_loader = load_dataset(dataset)

    if args.mode.lower() == 'train':

        model = train(train_loader, val_loader)

    elif args.mode.lower() == 'test':

        model = DaZhuangNet().to(device)

        optimizer = optim.Adam(model.parameters())

        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
        print(model)
        test(model, test_loader)

    else:
        raise RuntimeError("\"{0}\" is not a valid choice for execution mode.".format(args.mode))
