import glob
import pandas as pd
import os
from scipy import misc
import numpy as np

class PreProcessor:

    def __init__(self, input_dir):
        self.input_dir = input_dir

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
