#!/usr/bin/env python3
import main
import numpy as np
from math import *



nn = main.NeuralNet (8, [4, 4, 2])
nn.input([0.6, 0.4, 0.6, 0.2, 0.2, 0.8, 0.3, 0.9])
nn.update()

print ("This is test!")
print ("number of layers = " + str(nn.nLayers))
print ("number of nodes = " + str(nn.nNodes))
print ("number of weights = " + str(nn.nWeights))
nn.output()

#print (nn.inp)
#print (nn.n)
#print (nn.b)
#print (nn.w)
