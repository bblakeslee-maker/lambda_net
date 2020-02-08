import argparse
import torch
import torch.utils.data as tud
import torch.nn.functional as tnf
import torchvision.transforms as tvt
import cv2
import os
import numpy
import tqdm
# import csv
import sklearn.metrics as skm
# import math
import configparser
import dataset.dataset as loader
from utility import lambdaShell
import getpass
import sys


def main():
    # Parse arguments and load specified modules
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Config file to use for training.')
    parser.add_argument('--modelName', type=str, required=True,
                        help='Save name for trained model.')
    parser.add_argument('--resultName', type=str, required=True,
                        help='Destination for result images and statistics.')
    parser.add_argument('--gpu', type=int, required=False, default=0,
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
    valTransforms = tvt.Compose([tvt.Resize(imgDimensions)])
    valData = loader.Dataset(config, 'val', 200, transformList=valTransforms)
    valLoader = tud.DataLoader(
        valData, batch_size=config['params'].getint('batchSize'),
        shuffle=False, num_workers=4, pin_memory=True)

    # Load encoder and decoder architectures
    net = lambdaShell.LambdaShell(config)
    net.load_state_dict(
        torch.load(os.path.join('models', args.modelName,
                                args.modelName + '.pth')))
    if torch.cuda.is_available():
        gpuName = torch.cuda.get_device_name(args.gpu)
        print('STATUS: Running on {} (GPU {})!'.format(gpuName, args.gpu))
        device = torch.device('cuda:' + str(args.gpu))
        net = net.to(device)
    else:
        print('ERROR: No GPU available! Running on CPU!')

    idNumber = 0
    destination = os.path.join('models', args.modelName, 'results',
                               args.resultName)
    if not os.path.exists(destination):
        os.makedirs(destination)
    confusionMatrix = None
    with torch.no_grad():
        for (_, dataBatch) in enumerate(tqdm.tqdm(valLoader)):
            net.encoder.eval()
            net.decoder.eval()
            # Unpack data batch object and load images/truth to GPU
            refImg, tgtImg, labels, roiMask, paths, height, width = dataBatch
            refImg = refImg.to(device)
            tgtImg = tgtImg.to(device)
            # Feedforward evaluation
            segOut = net(refImg, tgtImg)
            # Process statistics
            segOut = torch.sigmoid(segOut).cpu()
            gridZeros = torch.zeros(config['preprocess'].getint('imgHeight'),
                                    config['preprocess'].getint('imgWidth'))
            gridOnes = torch.ones(config['preprocess'].getint('imgHeight'),
                                  config['preprocess'].getint('imgWidth'))
            predictions = torch.where(segOut < 0.5, gridZeros, gridOnes)
            predScaled = predictions * 255
            predScaled = predScaled.repeat(1, 3, 1, 1)
            labels = labels.cpu()
            expandLabels = torch.unsqueeze(labels, 1)
            expandLabels = expandLabels.repeat(1, 3, 1, 1)
            for i in range(predScaled.size()[0]):
                # Read images
                rawRefImg = cv2.imread(paths['ref'][i])
                rawTgtImg = cv2.imread(paths['tgt'][i])
                rawGtImg = cv2.imread(paths['gt'][i])
                rawPredImg = predScaled[i].cpu().numpy().transpose(
                    1, 2, 0)
                rawPredImg = cv2.resize(rawPredImg, (width[i], height[i]))
                # Concatenate
                top = numpy.concatenate((rawRefImg, rawTgtImg), axis=1)
                bottom = numpy.concatenate((rawGtImg, rawPredImg), axis=1)
                whole = numpy.concatenate((top, bottom), axis=0)
                # Write file
                sampleName = str(idNumber).zfill(5) + '.jpg'
                if not os.path.exists(destination):
                    os.makedirs(destination)
                cv2.imwrite(os.path.join(destination, sampleName), whole)
                # Compute statistics
                maskList = roiMask[i].numpy().reshape(-1)
                truthList = labels[i].byte().numpy().reshape(-1)
                predList = predictions[i].byte().cpu().numpy().reshape(-1)
                maskedTruth = truthList[maskList]
                maskedPred = predList[maskList]
                sampleConfusionMatrix = skm.confusion_matrix(
                    y_true=maskedTruth, y_pred=maskedPred, labels=[0, 1])
                if confusionMatrix is None:
                    confusionMatrix = sampleConfusionMatrix
                else:
                    confusionMatrix += sampleConfusionMatrix
                idNumber += 1
        gRecall = recall(confusionMatrix)
        gSpecificity = specificity(confusionMatrix)
        gfpr = falsePositiveRate(confusionMatrix)
        gfnr = falseNegativeRate(confusionMatrix)
        gpwc = percentWrongClass(confusionMatrix)
        gF1 = f1Score(confusionMatrix)
        gPrecision = precision(confusionMatrix)
        gChangeIou = iou(confusionMatrix)
        with open(os.path.join(destination, 'globalStats.txt'), 'w') as f:
            f.write('recall = {}\n'.format(gRecall))
            f.write('specificity = {}\n'.format(gSpecificity))
            f.write('FPR = {}\n'.format(gfpr))
            f.write('FNR = {}\n'.format(gfnr))
            f.write('PWC = {}\n'.format(gpwc))
            f.write('f1Score = {}\n'.format(gF1))
            f.write('precision = {}\n'.format(gPrecision))
            f.write('meanChangeIou = {}\n'.format(gChangeIou))


def meanSquareError(prediction, groundTruth):
    predSqueeze = torch.squeeze(prediction)
    mse = torch.sum((predSqueeze.type(torch.FloatTensor) - groundTruth.type(
        torch.FloatTensor)) ** 2)
    return mse.item()


def iou(confusionMatrix):
    truePos = getTruePos(confusionMatrix)
    falseNeg = getFalseNeg(confusionMatrix)
    falsePos = getFalsePos(confusionMatrix)
    iou = truePos / (truePos + falseNeg + falsePos)
    return iou


def recall(confusionMatrix):
    truePos = getTruePos(confusionMatrix)
    falseNeg = getFalseNeg(confusionMatrix)
    rec = truePos / (truePos + falseNeg)
    return rec


def specificity(confusionMatrix):
    trueNeg = getTrueNeg(confusionMatrix)
    falsePos = getFalsePos(confusionMatrix)
    spec = trueNeg / (trueNeg + falsePos)
    return spec


def falsePositiveRate(confusionMatrix):
    falsePos = getFalsePos(confusionMatrix)
    trueNeg = getTrueNeg(confusionMatrix)
    fpr = falsePos / (falsePos + trueNeg)
    return fpr


def falseNegativeRate(confusionMatrix):
    falseNeg = getFalseNeg(confusionMatrix)
    truePos = getTruePos(confusionMatrix)
    fnr = falseNeg / (truePos + falseNeg)
    return fnr


def precision(confusionMatrix):
    truePos = getTruePos(confusionMatrix)
    falsePos = getFalsePos(confusionMatrix)
    prec = truePos / (truePos + falsePos)
    return prec


def percentWrongClass(confusionMatrix):
    truePos = getTruePos(confusionMatrix)
    trueNeg = getTrueNeg(confusionMatrix)
    falseNeg = getFalseNeg(confusionMatrix)
    falsePos = getFalsePos(confusionMatrix)
    denom = truePos + falseNeg + falsePos + trueNeg
    pwc = (100 * (falseNeg + falsePos)) / denom
    return pwc


def f1Score(confusionMatrix):
    precis = precision(confusionMatrix)
    rec = recall(confusionMatrix)
    f1 = (2 * precis * rec) / (precis + rec)
    return f1


def getTrueNeg(confusionMatrix):
    return confusionMatrix[0][0]


def getTruePos(confusionMatrix):
    return confusionMatrix[1][1]


def getFalsePos(confusionMatrix):
    return confusionMatrix[0][1]


def getFalseNeg(confusionMatrix):
    return confusionMatrix[1][0]


if __name__ == '__main__':
    main()
