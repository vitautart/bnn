#!/usr/bin/env python3
import main
import numpy as np
from math import *

print ("This is testing!\n")

nn = main.NeuralNet (8, [4, 2])
nn.input([0.6, 0.4, 0.6, 0.2, 0.2, 0.8, 0.3, 0.9])
nn.update()

#print(main.AuxiliaryMath.sigmoid(np.array([1.3, 0.2, 1.6])))
print ("number of layers = " + str(nn.nLayers))
print (nn.nNodes)
print (nn.nWeights)
print (nn.inp)
print (nn.n)
#print (nn.b)
#print (nn.w)
