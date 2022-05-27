# Import
from pickle import TRUE
from pickletools import optimize
from pkgutil import ImpImporter
from turtle import color
from numpy import array
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os 
import datetime
import matplotlib.pyplot as plt

import argparse

from zmq import device

from model import *
from utils import progress_bar, draw_fig_2_data



# print('Cuda ready',torch.cuda.is_available())

if __name__ == "__main__":
    datetime_train = datetime.datetime.now()
    epoch_plush = 2
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume','-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device use: "+ device)
    best_acc = 0 #best test accurancy
    start_epoch = 0 # start from epoch 0 or last checkpoin epoch

    #Refresh cache GPU
    print('==> Cleanning Cache...')
    torch.cuda.empty_cache()

    # Data
    print('==> Preparing data...')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4811, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=2)
    
    # Class
    classes = ('plane','car','bird','cat','deer',
            'dog','frog','horse','ship','truck')

    # Model
    print('==> Building model...')
    # net = ResNet50(img_channels=3,num_classes=10)
    # net = RexNeXt29_4x64d()
    net = DenseNet201()
    net = net.to(device)

    if device =='cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    #Use check point
    args.resume = False
    print("Want to resume model", args.resume)
    if args.resume:
        #Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = args.lr, 
                        momentum = 0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training

    def train(epoch):
        print(f'Epoch:{epoch}')
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc_train = 100.*correct/total
        loss_train = train_loss/(batch_idx+1)

        return acc_train, loss_train
            
            
            
    # Test
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            acc_test = 100.*correct/total
            loss_test = test_loss/(batch_idx+1)

                

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('./checkpoint'):
                os.mkdir('./checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        
        return acc_test, loss_test

#Run 
    acc_array_train=[]
    acc_array_test=[]   
    loss_array_train=[]     
    loss_array_test=[] 
    for epoch in range(start_epoch, start_epoch+epoch_plush ):
        acc_train, loss_train= train(epoch)
        acc_test, loss_test = test(epoch)

        acc_array_train.append(acc_train)
        acc_array_test.append(acc_test)
        loss_array_train.append(loss_train)
        loss_array_test.append(loss_train)
        scheduler.step()
    state = {
        'Acc_train': acc_array_train,
        'Loss_train': loss_array_train,
        'Acc_test' : acc_array_test,
        'Loss_test' : loss_array_test
        }
    if not os.path.isdir('./Code/Acc_Loss'):
        os.mkdir('./Acc_Loss')
    torch.save(state,'./Acc_Loss/check.pth')

#Showdata train 
    draw_fig_2_data(acc_array_test, acc_array_train, start_epoch, epoch_plush,'Epoch','Accurency (%)','./Acc_Loss/fig/acc.png')
    # index = [a for a in range(start_epoch, start_epoch+epoch_plush,1)]
    # fig, ax = plt.subplots()
    # ax.plot(index,acc_array_train, color='red', label='Train')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accurency (%)')
    # ax.plot(index,acc_array_test, color='blue', label='Test')
    # ax.legend()
    # plt.show()
    # plt.savefig('./Code/Acc_Loss/fig/acc.png')

#Showdata loss
    draw_fig_2_data(loss_array_test, loss_array_train, start_epoch, epoch_plush,'Epoch','Loss','./Acc_Loss/fig/loss.png')
    # fig, ax = plt.subplots()
    # ax.plot(index,loss_array_train, color='red', label='Train')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accurency (%)')
    # ax.plot(index,loss_array_test, color='blue', label='Test')
    # ax.legend()
    # plt.show()
    # plt.savefig('./Code/Acc_Loss/fig/loss.png')



