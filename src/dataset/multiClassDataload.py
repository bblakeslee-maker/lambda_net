import torch
import torch.utils.data.dataset as tudd
import cv2
import os
import PIL
import numpy as np


class Dataset(tudd.Dataset):
    def __init__(self, config, splitType, transformList=None):
        self.__dataPath = config['paths']['dataPath']
        self.__splitFile = config['paths'][splitType + 'Split']
        self.__splitType = splitType
        self.__transformList = transformList
        self.__config = config
        self.__dataEntries = self.__readSplitFile()

    def __readSplitFile(self):
        truthList = list()
        with open(self.__splitFile, 'r') as f:
            for line in f:
                t0Path, t1Path, gtPath = line.strip().split(',')
                truthList.append((t0Path, t1Path, gtPath))
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
        t0RelPath, t1RelPath, gtRelPath = self.__dataEntries[idx]
        t0Path = os.path.join(self.__dataPath, t0RelPath)
        t1Path = os.path.join(self.__dataPath, t1RelPath)
        gtPath = os.path.join(self.__dataPath, gtRelPath)
        paths = {'t0': t0Path, 't1': t1Path, 'gt': gtPath}
        t0Img = PIL.Image.open(t0Path)
        t1Img = PIL.Image.open(t1Path)
        gtImg = PIL.Image.open(gtPath)
        width, height = t0Img.size
        if self.__transformList is not None:
            t0Img = self.__transformList(t0Img)
            t1Img = self.__transformList(t1Img)
            gt0Img = self.__transformList(gtImg)
        t0Img = self.__meanCenterData(t0Img)
        t1Img = self.__meanCenterData(t1Img)

        # Cast to array and rescale to class values
        # Class Values:
        # No Change -- 0
        # Additive Change -- 1
        # Subtractive Change -- 2
        # Exchange -- 3
        gtImg = np.array(gt0Img, dtype=np.uint8) / 85
        dirGtImg = torch.from_numpy(gtImg).long()

        if self.__splitType == 'train':
            return t0Img, t1Img, dirGtImg, paths, int(height), int(width)
        elif self.__splitType == 'val':
            category, video, *_ = t0RelPath.split('/')
            roiPath = os.path.join(self.__dataPath, category, video, 'ROI.bmp')
            paths['roi'] = roiPath
            roiMask = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)
            maskWidth = self.__config['preprocess'].getint('imgWidth')
            maskHeight = self.__config['preprocess'].getint('imgHeight')
            roiMask = cv2.resize(roiMask, (maskWidth, maskHeight))
            roiMask = roiMask.astype(bool)
            return t0Img, t1Img, dirGtImg, roiMask, paths, int(height), int(width)

    def __len__(self):
        return len(self.__dataEntries)
