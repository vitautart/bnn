#!/usr/bin/env python3
import numpy as np
from mnist_parser import *
from PIL import Image

mp = mnistParser ('../mnist/t10k-labels.idx1-ubyte', '../mnist/t10k-images.idx3-ubyte')
mp.open()

labels = mp.parse_labels()

'''img = mp.parse_img_2()
arr = np.uint8(img[1])
img = Image.fromarray(arr, mode = 'L')
img.show()'''

img2 = mp.parse_img()[20]
print (labels[20])
#print(img2)
arr = np.zeros ((mp.imfrows, mp.imfcoll), dtype = 'uint8')
for i in range(mp.imfrows):
    for j in range(mp.imfcoll):
        arr[i][j] = int(img2[i*mp.imfcoll+j]*255.0)
img = Image.fromarray(arr, mode = 'L')
img.show()

mp.close()


#print (list([x , y] for [x, y] in [labels, img2]))
#print(mp.lfnumber)
print(mp.imfmagic)
print(mp.imfnumber)
print(mp.imfrows)
print(mp.imfcoll)
#print(labels)
