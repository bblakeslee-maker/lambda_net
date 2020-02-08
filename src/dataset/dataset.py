import torch
import torch.utils.data.dataset as tudd
import cv2
import os
import PIL
import numpy as np


class Dataset(tudd.Dataset):
    def __init__(self, config, splitType, threshold, transformList=None):
        self.__dataPath = config['paths']['dataPath']
        self.__splitFile = config['paths'][splitType + 'Split']
        self.__splitType = splitType
        self.__threshold = threshold
        self.__transformList = transformList
        self.__config = config
        self.__dataEntries = self.__readSplitFile()

    def __readSplitFile(self):
        truthList = list()
        with open(self.__splitFile, 'r') as f:
            for line in f:
                refPath, tgtPath, gtPath = line.strip().split(',')
                truthList.append((refPath, tgtPath, gtPath))
        return truthList

    def __meanCenterData(self, img):
        imgTmp = np.array(img, dtype=np.uint8)
        imgTmp = imgTmp.astype(np.float64)
        redMean = self.__config['preprocess'].getfloat('redMean')
        greenMean = self.__config['preprocess'].getfloat('greenMean')
        blueMean = self.__config['preprocess'].getfloat('blueMean')
        mean = (redMean, greenMean, blueMean)
        imgTmp -= mean
        imgTmp = imgTmp.transpose(2, 0, 1)
        return torch.from_numpy(imgTmp).float()

    def __getitem__(self, idx):
        relRefPath, relTgtPath, relGtPath = self.__dataEntries[idx]
        refPath = os.path.join(self.__dataPath, relRefPath)
        tgtPath = os.path.join(self.__dataPath, relTgtPath)
        gtPath = os.path.join(self.__dataPath, relGtPath)
        paths = {'ref': refPath, 'tgt': tgtPath, 'gt': gtPath}
        refImg = PIL.Image.open(refPath)
        tgtImg = PIL.Image.open(tgtPath)
        gtImg = PIL.Image.open(gtPath)
        width, height = refImg.size
        if self.__transformList is not None:
            refImg = self.__transformList(refImg)
            tgtImg = self.__transformList(tgtImg)
            gtImg = self.__transformList(gtImg)
        refImg = self.__meanCenterData(refImg)
        tgtImg = self.__meanCenterData(tgtImg)
        gtImg = np.array(gtImg, dtype=np.uint8)
        gtImg = torch.from_numpy(gtImg).long()
        gtImg[gtImg < self.__threshold] = 0
        gtImg[gtImg >= self.__threshold] = 1
        if self.__splitType == 'train':
            return refImg, tgtImg, gtImg, paths, int(height), int(width)
        elif self.__splitType == 'val':
            category, video, *_ = relRefPath.split('/')
            roiPath = os.path.join(self.__dataPath, category, video, 'ROI.bmp')
            paths['roi'] = roiPath
            roiMask = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)
            maskWidth = self.__config['preprocess'].getint('imgWidth')
            maskHeight = self.__config['preprocess'].getint('imgHeight')
            roiMask = cv2.resize(roiMask, (maskWidth, maskHeight))
            roiMask = roiMask.astype(bool)
            return refImg, tgtImg, gtImg, roiMask, paths, int(height), int(width)

    def __len__(self):
        return len(self.__dataEntries)
