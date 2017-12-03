#!/usr/bin/env python3
import numpy as np
from math import *

class AuxiliaryMath:

    #vector sigmoid function
    @staticmethod
    def sigmoid (x):
        g = np.vectorize (lambda x: 1/(1+exp(-x)))
        return g(x)

    #first derivative of vector sigmoid function
    @staticmethod
    def dsigmoid (x):
        sigm = lambda x: 1/(1+exp(-x))
        g = np.vectorize (lambda x: sigma(x)*(1-sigma(x)))
        return g(x)

class InputReader:
    def __init__ (self):
        return 0

class Teacher:
    def __init__ (self):
        return 0

class NeuralNet:
    def __init__ (self, inputQuantity, nodesOnLayers): #inputQuantity - int, nodesOnLayers - [int, int, .., int]
        self.nNodes = 0
        self.nWeights = 0
        self.nLayers = 0
        self.inp = None # input vector (input nodes)
        self.n = []  # list by layers of nodes vectors
        self.b = []  # list by layers of biases vectors
        self.w = [] # list by layers of weights matrices
        self.layersIndex = []
        if isinstance(nodesOnLayers, list) and isinstance(inputQuantity, int):
            if all(isinstance(ins, int) for ins in nodesOnLayers):
                self.nNodes = inputQuantity
                self.nLayers = len(nodesOnLayers)
                self.layersIndex = range(self.nLayers)
                self.inp = np.zeros (inputQuantity)
                quantityOnPrevious = None
                for i in self.layersIndex:
                    if i==0:
                        quantityOnPrevious = inputQuantity
                    else:
                        quantityOnPrevious = nodesOnLayers[i-1]
                    self.nNodes += nodesOnLayers[i]
                    self.nWeights = self.nWeights + nodesOnLayers[i]*quantityOnPrevious
                    self.n.append (np.zeros (nodesOnLayers[i], dtype=np.float)) #initialization of nodes on each layers
                    self.b.append (np.random.rand(nodesOnLayers[i])) #initialization of biases on each layers
                    self.w.append(np.random.rand(nodesOnLayers[i], quantityOnPrevious)) #initialization of weights on each layers
            else:
                print ("Quantity of nodes must be integer")
        else:
            print ("Please enter neural network construction in following format (numberNodesOnInput, [numberNodesOn1Layer, numberNodesOn2Layer, ... , numberNodesOnNLayer])")

    def input (self, inputParameters):
        if (len(inputParameters) == self.inp.size):
            self.inp = np.array (inputParameters)
        else:
            print ("Please put correct quantity of input values")

    def update (self):
        for i in self.layersIndex:
            if i==0:
                self.n[i] = AuxiliaryMath.sigmoid(self.w[i].dot(self.inp)+self.b[i])
            else:
                self.n[i] = AuxiliaryMath.sigmoid(self.w[i].dot(self.n[i-1])+self.b[i])

    def output (self):
        print (self.n[self.nLayers-1])
        return self.n[self.nLayers-1]
