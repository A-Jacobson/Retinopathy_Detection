import glob
import pandas as pd
import os
import numpy as np
import cv2
from multiprocessing import Pool
from scipy import misc
import itertools


def get_channel_means(images):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    n = float(len(images))
    for im in images:
        im = misc.imread(im)
        sum_r += np.mean(im[:, :, 0])
        sum_g += np.mean(im[:, :, 1])
        sum_b += np.mean(im[:, :, 2])
    mean_r = sum_r / n
    mean_g = sum_g / n
    mean_b = sum_b / n
    return np.array([mean_r, mean_g, mean_b], dtype='uint8')

def get_mean(images):
    total = 0
    n = float(len(images))
    for im in images:
        im = misc.imread(im)
        total += np.mean(im)
    mean = total / n
    return np.array(mean, dtype='uint8')

def get_sample_matrix(images):
    return np.array([misc.imread(im) for im in images])


class ImagePreProcessor:

    def smart_crop(self, im, threshold=30):
        """
        Detects and crops around eyes in each image
        im = image as np array
        threshold = min pixel value for binary-otsu thresholding
        """
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
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

    def norm(self, im, method='channel', mean=59, mean_r=81, mean_g=57, mean_b=41, mean_image=None):
        im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if method == 'basic':
            im -= mean
        elif method == 'channel':
            im[:,:,0] -= mean_r
            im[:,:,1] -= mean_g
            im[:,:,2] -= mean_b
        elif method == 'image':
            im -= mean_image
        return im

    def preprocess_img(self, im, size=(512, 512), threshold=50, hist_equalize=False, norm=None):
        im = cv2.imread(im)
        im = self.smart_crop(im, threshold)
        im = self.smart_resize(im, size)
        if hist_equalize == True:
            im = self.histogram_equalization(im)
        if norm == 'basic':
            return self.norm(im, method='basic')
        elif norm == 'channel':
            return self.norm(im, method='channel')
        else:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def preprocess_directory(self, input_dir, output_dir, size=(512, 512), threshold=50, norm=False):
        """process all images in input dir and save to output_dir"""
        files = os.listdir(input_dir)
        for f in files:
            inpath = os.path.join(input_dir, f)
            outpath = os.path.join(output_dir, f)
            im = self.preprocess_img(inpath, norm=norm, size=size, threshold=threshold)
            misc.imsave(outpath, im)
