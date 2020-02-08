import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torchvision.transforms as tvt
import os
import random
import configparser
import dataset.dataset as loader
from utility import lambdaShell
import time
import datetime
import shutil
import getpass
import sys


def main():
    startTime = time.time()
    # Parse arguments and load specified modules
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Config file to use for training.')
    parser.add_argument('--modelName', type=str, required=True,
                        help='Save name for trained model.')
    parser.add_argument('--gpu', type=int, required=False, default=-1,
                        help='Address of GPU to run on.')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join('cfgs', args.config + '.txt'))

    # Process path information
    username = getpass.getuser()
    if '{}' in config['paths']['dataPath']:
        print('STATUS: Inserting username {} into data path!'.format(username))
        config['paths']['dataPath'] = config['paths']['dataPath'].format(username)
        print('STATUS: Data path = {}'.format(config['paths']['dataPath']))
    if '{}' in config['paths']['trainSplit']:
        print('STATUS: Inserting username {} into training split path!'.format(username))
        config['paths']['trainSplit'] = config['paths']['trainSplit'].format(username)
        print('STATUS: Train split path = {}'.format(config['paths']['trainSplit']))
    if '{}' in config['paths']['valSplit']:
        print('STATUS: Inserting username {} into val split path!'.format(username))
        config['paths']['valSplit'] = config['paths']['valSplit'].format(username)
        print('STATUS: Val split path = {}'.format(config['paths']['valSplit']))
    if '{}' in config['paths']['testSplit']:
        print('STATUS: Inserting username {} into test split path!'.format(username))
        config['paths']['testSplit'] = config['paths']['testSplit'].format(username)
        print('STATUS: Test split path = {}'.format(config['paths']['testSplit']))

    # Construct transforms and training and validation datasets
    imgDimensions = (config['preprocess'].getint('imgWidth'),
                     config['preprocess'].getint('imgHeight'))
    trainTransforms = tvt.Compose([tvt.Resize(imgDimensions)])
    trainData = loader.Dataset(config, 'train', 200,
                               transformList=trainTransforms)
    trainLoader = tud.DataLoader(
        trainData, batch_size=config['params'].getint('batchSize'),
        shuffle=True, num_workers=4, pin_memory=True)

    # Configure destination folder
    os.makedirs(os.path.join('models', args.modelName), exist_ok=True)
    shutil.copy(os.path.join('cfgs', args.config + '.txt'),
                os.path.join('models', args.modelName, args.config + '.txt'))

    # Load network
    net = lambdaShell.LambdaShell(config)
    # Transfer learning loader
    if config.has_option('arch', 'baseTrainModel'):
        print('STATUS: Loading pretrained model ' +
              config['arch']['baseTrainModel'] + '!')
        net.load_state_dict(
            torch.load('models/' + config['arch']['baseTrainModel'] + '.pth'))
    else:
        print('STATUS: Training model from scratch!')

    # CUDA configuration code
    if torch.cuda.is_available():
        if args.gpu == -1:
            print('STATUS: Running on any GPU!')
            device = torch.device('cuda')
        else:
            gpuName = torch.cuda.get_device_name(args.gpu)
            print('STATUS: Running on {} (GPU {})!'.format(gpuName, args.gpu))
            device = torch.device('cuda:' + str(args.gpu))
        net = net.to(device)
    else:
        print('ERROR: No GPU available! Running on CPU!')

    # Configure loss
    changeWeight = torch.Tensor([config['params'].getfloat('changeWeight')])
    if changeWeight < 0:
        print('STATUS: Running unweighted BCE loss!')
        lossCriteria = nn.BCEWithLogitsLoss().to(device)
    else:
        print('STATUS: Running weighted BCE loss with weight {}!'.format(changeWeight))
        lossCriteria = nn.BCEWithLogitsLoss(pos_weight=changeWeight).to(device)

    # Configure optimizer
    if config['arch']['encoderModel'] == 'vgg16':
        print('STATUS: Training decoder only!')
        optimizer = optim.Adam(net.decoder.parameters(),
                               lr=config['params'].getfloat('learningRate'))
    else:
        print('STATUS: Training entire model!')
        optimizer = optim.Adam(net.parameters(),
                               lr=config['params'].getfloat('learningRate'))

    # Configure learning rate scheduler
    if config['params'].getint('decayInterval') != 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config['params'].getint('decayInterval'),
            gamma=config['params'].getfloat('decayRate'))

    # Do training
    for epoch in range(config['params'].getint('numEpochs')):
        if config['arch']['encoderModel'] == 'vgg16':
            net.encoder.eval()
            net.decoder.train()
        else:
            net.train()
        for (batchIdx, dataBatch) in enumerate(trainLoader):
            step = epoch * len(trainLoader) + batchIdx
            # Unpack data batch object and load images/truth to GPU
            refImg, tgtImg, labels, paths, height, width = dataBatch
            if config['params'].getboolean('randomInputSwap'):
                flag = random.randint(0, 1)
                if flag == 1:
                    refImg, tgtImg = tgtImg, refImg
            refImg = refImg.to(device)
            tgtImg = tgtImg.to(device)
            labels = labels.to(device)
            # Feedforward evaluation
            segOut = net(refImg, tgtImg)
            # Compute loss and perform optimization step
            segOut = torch.squeeze(segOut)
            segOut = segOut.float()
            labels = torch.squeeze(labels)
            labels = labels.float()
            loss = lossCriteria(segOut, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config['params'].getint('decayInterval') != 0:
                scheduler.step(epoch)
            # Report training
            if batchIdx % 20 == 0:
                print('{}: Epoch [{} / {}] Index: [{} / {}] Loss: {}'.format(
                    args.modelName, epoch + 1,
                    config['params'].getint('numEpochs'), batchIdx,
                    len(trainLoader), loss))
                lossEntry = {'epoch': epoch,
                             'batchIdx': batchIdx,
                             'numIdx': len(trainLoader),
                             'loss': loss.item()}
                lossLog = os.path.join(
                    'models', args.modelName, args.modelName + '_loss.log')
                with open(lossLog, 'a') as f:
                    json.dump(lossEntry, f)
                    f.write('\n')
        print('STATUS: Saving {} epoch {}!'.format(
            args.modelName, epoch + 1))
        torch.save(net.state_dict(), 'models/checkpoint.pth')
    torch.save(net.state_dict(),
               os.path.join('models', args.modelName, args.modelName + '.pth'))
    endTime = time.time()
    elapsedTime = str(datetime.timedelta(seconds=(endTime - startTime)))
    print('Total training time: {}'.format(elapsedTime))


if __name__ == '__main__':
    main()
