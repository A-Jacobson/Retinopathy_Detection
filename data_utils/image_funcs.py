import glob
import pandas as pd
import os
import numpy as np
import cv2
from multiprocessing import Pool
from scipy import misc
import itertools


class ImageGenerator:
    """
    warning, current implimentation does not work
    """
    def __init__(self, X_path=os.path.join("E:", "DR_Data", "Train_512"), y_path=os.path.join('data', 'trainLabels.csv'), shape=(3, 512, 512), ending=".jpeg"):
        self.X_path = X_path
        self.y_path = y_path
        self.shape = shape
        self.ending = ending

    def reshape(self, im, shape):
        im = misc.imread(im)
        return np.reshape(im, shape)

    def x_generator(self, directory):
        for image in glob.glob(os.path.join(directory, '*'+self.ending)):
            yield self.reshape(image, self.shape)

    def y(self, directory):
        y = pd.read_csv(directory)
        return y.level.values

    def get_batch(self, batch_size=5):
        x_gen = self.x_generator(self.X_path)
        y = self.y(self.y_path)
        X_batch = np.array(list(itertools.islice(x_gen, 0, batch_size, 1)))
        y_batch = np.array(list(itertools.islice(y, 0, batch_size, 1)))
        return X_batch, y_batch


class ImagePreProcessor:

    def smart_crop(self, im, threshold=30):
        """
        Detects and crops around eyes in each image
        im = image as np array
        threshold = min pixel value for binary-otsu thresholding
        """
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return im[y:y+h,x:x+w]

    def smart_resize(self, im, size=(512, 512)):
        """
        resize an image while retaining aspect ratio, pad with black background
        im = image as np array
        """
        h, w, _ = im.shape

        if w > h:
            difference = w - h
            top = int((difference/2 + (difference%2)))
            bottom = int(np.ceil(difference/2.))
            left = 0
            right = 0
            im = cv2.copyMakeBorder(im, top, bottom, left, right,cv2.BORDER_CONSTANT)
        if h > w:
            difference = h - w
            top = 0
            bottom = 0
            left = int((difference/2 + (difference%2)))
            right = int(np.ceil(difference/2.))
            im = cv2.copyMakeBorder(im, top, bottom, left, right,cv2.BORDER_CONSTANT)
        im = cv2.resize(im, size)
        return im

    def histogram_equalization(self, im):
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        im_yuv[:,:,0] = cv2.equalizeHist(im[:,:,0])
        return cv2.cvtColor(im_yuv, cv2.COLOR_YUV2RGB)

    def preprocess_img(self, im, size=(512, 512), threshold=50, norm=False):
        im = cv2.imread(im)
        im = self.smart_crop(im, threshold)
        im = self.smart_resize(im, size)
        if norm == True:
            return self.histogram_equalization(im)
        else:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def preprocess_directory(self, input_dir, output_dir, size=(512, 512), threshold=50, norm=False):
        """process all images in input dir and save to output_dir"""
        files = os.listdir(input_dir)
        for f in files:
            inpath = os.path.join(input_dir, f)
            outpath = os.path.join(output_dir, f)
            im = self.preprocess_img(inpath, norm=norm)
            misc.imsave(outpath, im)
