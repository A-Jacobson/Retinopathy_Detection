import cv2
import os
import numpy as np

im = cv2.imread(os.path.join('data', 'samples', '10_left.jpeg'))

detector = cv2.SimpleBlobDetector()

keypoints = detector.detect(im)

print keypoints
