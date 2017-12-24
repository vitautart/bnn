#!/usr/bin/env python3
import main
import numpy as np
import mnist_parser
from math import *
from PIL import Image

testNet = main.NeuralNet (784, [30, 10])
testNet.importNet("bigTestNet784_30_10.txt")
dataParser = mnist_parser.mnistParser ('../mnist/train-labels.idx1-ubyte', '../mnist/train-images.idx3-ubyte')
#dataParser = mnist_parser.mnistParser ('../mnist/t10k-labels.idx1-ubyte', '../mnist/t10k-images.idx3-ubyte')
dataParser.open()
labels = dataParser.parse_labels()
images = dataParser.parse_img()
dataParser.close()

testNet.train(labels, images, 1000, 10, 0.5)

testNet.exportNet("bigTestNet784_30_10.txt")


#testing
mp = mnist_parser.mnistParser ('../mnist/t10k-labels.idx1-ubyte', '../mnist/t10k-images.idx3-ubyte')
mp.open()
lbl = mp.parse_labels()[100]
image_vector = mp.parse_img()[100]
'''
arr = np.zeros ((mp.imfrows, mp.imfcoll), dtype = 'uint8')
for i in range(mp.imfrows):
    for j in range(mp.imfcoll):
        arr[i][j] = int(image_vector[i*mp.imfcoll+j]*255.0)
img = Image.fromarray(arr, mode = 'L')
img.show()
'''
mp.close()

testNet.input(image_vector)
testNet.update()
#testNet.backpropagate(lbl)
#testNet.push(100000)
print('Input value')
print (lbl)
print ("Error")
print (testNet.verify(image_vector, lbl))
print ("Given value")
print(testNet.output())
#print(testNet.delta)
#print (lbl)
