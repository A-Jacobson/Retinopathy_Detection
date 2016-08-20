import glob
import pandas as pd
import os
from scipy import misc
import numpy as np
import cv2

class PreProcessor:


    def resize_directory(input_dir, output_dir, size=(512, 512, 3)):
        files = os.listdir(input_dir)
        for f in files:
            inpath = os.path.join(input_dir, f)
            outpath = os.path.join(output_dir, f)
            img = misc.imread(inpath)
            img = misc.imresize(img, size=size)
            misc.imsave(outpath, img)

    def matrix_from_dir(directory, name='X_train'):
        images = glob.glob(os.path.join(output_dir, "*"))
        X = np.array([misc.imread(i) for i in images])
        X = X.reshape(X.shape[0], 3, 512, 512)
        np.save(os.path.join('..', 'data', 'X_train'), X)

def smart_crop(im, threshold=30):
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

def smart_resize(im, size=(512, 512)):
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

def histogram_equalization(im):
    img_yuv = cv2.cvtColor(test, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def preprocess_img(im, size=(512, 512), threshold=50):
    im = cv2.imread(im)
    im = crop(im, threshold)
    im = smart_resize(im, size)
    return im

def to_training_shape(im):
    return im.reshape(3, 512, 512)



from multiprocessing import Pool
import os

path1 = "some/path"
path2 = "some/other/path"

listing = os.listdir(path1)

p = Pool(5) # process 5 images simultaneously

def resize_file(path, size=(512, 512, 3)):
    img = misc.imread(inpath)
    img = misc.imresize(img, size=size)
    im = Image.open(path1 + path)
    im.resize((50,50))                # need to do some more processing here
    im.save(os.path.join(path2,path), "JPEG")

p.map(process_fpath, listing)
