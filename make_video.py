
import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('run1/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
length =20 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('run1.mp4', fourcc, length, size) 
#out = cv2.VideoWriter('run1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 48, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()