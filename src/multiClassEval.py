import argparse
import torch
import torch.utils.data as tud
import torchvision.transforms as tvt
import cv2
import os
import numpy
import tqdm
# import csv
import sklearn.metrics as skm
# import math
import configparser
import dataset.multiClassDataload as loader
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
    valData = loader.Dataset(config, 'val', transformList=valTransforms)
    valLoader = tud.DataLoader(
        valData, batch_size=config['params'].getint('batchSize'),
        shuffle=False, num_workers=4, pin_memory=True)

    # Load encoder and decoder architectures
    net = lambdaShell.LambdaShell(config)
    net.load_state_dict(torch.load(
        os.path.join('models', args.modelName, args.modelName + '.pth')))
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
            t0Img, t1Img, labels, roiMask, paths, height, width = dataBatch
            t0Img = t0Img.to(device)
            t1Img = t1Img.to(device)
            # Feedforward evaluation
            segOut = net(t0Img, t1Img)
            # Process statistics
            _, predictions = segOut.max(dim=1, keepdim=True)
            predictions = torch.squeeze(predictions)
            print('predictions.shape = {}'.format(predictions.shape))
            labels = labels.cpu()
            for i in range(predictions.size()[0]):
                # Read images
                t0RawImg = cv2.imread(paths['t0'][i])
                t1RawImg = cv2.imread(paths['t1'][i])
                singleLabel = labels[i].cpu().numpy().astype(numpy.uint8)
                singlePred = predictions[i].cpu().numpy().astype(numpy.uint8)
                # Colorize
                singleLabel = classToImg(singleLabel)
                singlePred = classToImg(singlePred)
                singleLabel = cv2.resize(singleLabel, (width[i], height[i]))
                singlePred = cv2.resize(singlePred, (width[i], height[i]))
                # Concatenate
                top = numpy.concatenate((t0RawImg, t1RawImg), axis=1)
                bottom = numpy.concatenate((singleLabel, singlePred), axis=1)
                whole = numpy.concatenate((top, bottom), axis=0)
                # Write file
                sampleName = str(idNumber).zfill(5) + '.png'
                if not os.path.exists(destination):
                    os.makedirs(destination)
                cv2.imwrite(os.path.join(destination, sampleName), whole)
                # Compute statistics
                maskList = roiMask[i].numpy().reshape(-1)
                truthList = labels[i].byte().numpy().reshape(-1)
                predList = predictions[i].byte().cpu().numpy().reshape(-1)
                maskedTruth = truthList[maskList]
                maskedPred = predList[maskList]
                sampleConfusionMatrix = skm.multilabel_confusion_matrix(
                    y_true=maskedTruth, y_pred=maskedPred, labels=[0, 1, 2, 3])
                if confusionMatrix is None:
                    confusionMatrix = sampleConfusionMatrix
                else:
                    confusionMatrix += sampleConfusionMatrix
                idNumber += 1
        gNcRec,  gAddRec,  gSubRec,  gExRec  = recall(confusionMatrix)
        gNcSpec, gAddSpec, gSubSpec, gExSpec = specificity(confusionMatrix)
        gNcFpr,  gAddFpr,  gSubFpr,  gExFpr  = falsePositiveRate(confusionMatrix)
        gNcFnr,  gAddFnr,  gSubFnr,  gExFnr  = falseNegativeRate(confusionMatrix)
        gNcPwc,  gAddPwc,  gSubPwc,  gExPwc  = percentWrongClass(confusionMatrix)
        gNcF1,   gAddF1,   gSubF1,   gExF1   = f1Score(confusionMatrix)
        gNcPrec, gAddPrec, gSubPrec, gExPrec = precision(confusionMatrix)
        gNcIou,  gAddIou,  gSubIou,  gExIou  = iou(confusionMatrix)
        with open(os.path.join(destination, 'globalStats.txt'), 'w') as f:
            f.write('ncRec = {}\n'.format(gNcRec))
            f.write('addRec = {}\n'.format(gAddRec))
            f.write('subRec = {}\n'.format(gSubRec))
            f.write('exRec = {}\n'.format(gExRec))
            f.write('ncSpec = {}\n'.format(gNcSpec))
            f.write('addSpec = {}\n'.format(gAddSpec))
            f.write('subSpec = {}\n'.format(gSubSpec))
            f.write('exSpec = {}\n'.format(gExSpec))
            f.write('ncFPR = {}\n'.format(gNcFpr))
            f.write('addFPR = {}\n'.format(gAddFpr))
            f.write('subFPR = {}\n'.format(gSubFpr))
            f.write('exFPR = {}\n'.format(gExFpr))
            f.write('ncFNR = {}\n'.format(gNcFnr))
            f.write('addFNR = {}\n'.format(gAddFnr))
            f.write('subFNR = {}\n'.format(gSubFnr))
            f.write('exFNR = {}\n'.format(gExFnr))
            f.write('ncPWC = {}\n'.format(gNcPwc))
            f.write('addPWC = {}\n'.format(gAddPwc))
            f.write('subPWC = {}\n'.format(gSubPwc))
            f.write('exPWC = {}\n'.format(gExPwc))
            f.write('ncF1Score = {}\n'.format(gNcF1))
            f.write('addF1Score = {}\n'.format(gAddF1))
            f.write('subF1Score = {}\n'.format(gSubF1))
            f.write('exF1Score = {}\n'.format(gExF1))
            f.write('ncPrec = {}\n'.format(gNcPrec))
            f.write('addPrec = {}\n'.format(gAddPrec))
            f.write('subPrec = {}\n'.format(gSubPrec))
            f.write('exPrec = {}\n'.format(gExPrec))
            f.write('ncIOU = {}\n'.format(gNcIou))
            f.write('addIOU = {}\n'.format(gAddIou))
            f.write('subIOU = {}\n'.format(gSubIou))
            f.write('exIOU = {}\n'.format(gExIou))


def classToImg(pred):
    # Temporary channel buffers
    width, height = pred.shape
    red = numpy.zeros((width, height, 1), dtype=numpy.uint8)
    green = numpy.zeros((width, height, 1), dtype=numpy.uint8)
    blue = numpy.zeros((width, height, 1), dtype=numpy.uint8)
    # Assign colors to classes
    red[pred == 1] = 255    # Additive Change (1)
    green[pred == 3] = 255  # Exchange (3)
    blue[pred == 2] = 255   # Subtractive Change (2)
    # Generate image
    img = numpy.concatenate((blue, green, red), axis=2)
    return img


def iou(confusionMatrix):
    iouList = list()
    for i in range(0, 4):
        truePos = getTruePos(confusionMatrix[i])
        falseNeg = getFalseNeg(confusionMatrix[i])
        falsePos = getFalsePos(confusionMatrix[i])
        iou = truePos / (truePos + falseNeg + falsePos)
        iouList.append(iou)
    return iouList


def recall(confusionMatrix):
    recallList = list()
    for i in range(0, 4):
        truePos = getTruePos(confusionMatrix[i])
        falseNeg = getFalseNeg(confusionMatrix[i])
        rec = truePos / (truePos + falseNeg)
        recallList.append(rec)
    return recallList


def specificity(confusionMatrix):
    specList = list()
    for i in range(0, 4):
        trueNeg = getTrueNeg(confusionMatrix[i])
        falsePos = getFalsePos(confusionMatrix[i])
        spec = trueNeg / (trueNeg + falsePos)
        specList.append(spec)
    return specList


def falsePositiveRate(confusionMatrix):
    fprList = list()
    for i in range(0, 4):
        falsePos = getFalsePos(confusionMatrix[i])
        trueNeg = getTrueNeg(confusionMatrix[i])
        fpr = falsePos / (falsePos + trueNeg)
        fprList.append(fpr)
    return fprList


def falseNegativeRate(confusionMatrix):
    fnrList = list()
    for i in range(0, 4):
        falseNeg = getFalseNeg(confusionMatrix[i])
        truePos = getTruePos(confusionMatrix[i])
        fnr = falseNeg / (truePos + falseNeg)
        fnrList.append(fnr)
    return fnrList


def precision(confusionMatrix):
    precList = list()
    for i in range(0, 4):
        truePos = getTruePos(confusionMatrix[i])
        falsePos = getFalsePos(confusionMatrix[i])
        prec = truePos / (truePos + falsePos)
        precList.append(prec)
    return precList


def percentWrongClass(confusionMatrix):
    pwcList = list()
    for i in range(0, 4):
        truePos = getTruePos(confusionMatrix[i])
        trueNeg = getTrueNeg(confusionMatrix[i])
        falseNeg = getFalseNeg(confusionMatrix[i])
        falsePos = getFalsePos(confusionMatrix[i])
        denom = truePos + falseNeg + falsePos + trueNeg
        pwc = (100 * (falseNeg + falsePos)) / denom
        pwcList.append(pwc)
    return pwcList


def f1Score(confusionMatrix):
    f1List = list()
    precis = precision(confusionMatrix)
    rec = recall(confusionMatrix)
    for i in range(0, 4):
        f1 = (2 * precis[i] * rec[i]) / (precis[i] + rec[i])
        f1List.append(f1)
    return f1List


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
