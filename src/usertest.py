#!/usr/bin/env python3
import main
import numpy as np
import mnist_parser
from math import *
from PIL import Image

testNet = main.NeuralNet (784, [30, 10])
testNet.importNet("bigTestNet784_30_10.txt")

sample_number = 502

mp = mnist_parser.mnistParser ('../mnist/t10k-labels.idx1-ubyte', '../mnist/t10k-images.idx3-ubyte')
mp.open()
lbl = mp.parse_labels()[sample_number]
image_vector = mp.parse_img()[sample_number]

arr = np.zeros ((mp.imfrows, mp.imfcoll), dtype = 'uint8')
for i in range(mp.imfrows):
    for j in range(mp.imfcoll):
        arr[i][j] = int(image_vector[i*mp.imfcoll+j]*255.0)
img = Image.fromarray(arr, mode = 'L')
img.show()
mp.close()

testNet.input(image_vector)
testNet.update()

print('Input value')
print (lbl)
print ("Error")
print (testNet.verify(image_vector, lbl))
print ("Given value")
print(testNet.output())
