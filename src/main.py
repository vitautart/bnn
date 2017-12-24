#!/usr/bin/env python3
import numpy as np
from math import *
import random

class AuxiliaryMath:

    #vector sigmoid function
    @staticmethod
    def sigmoid (x):
        g = np.vectorize (lambda x: 1/(1+exp(-x)))
        return g(x)

    #first derivative of vector sigmoid function
    @staticmethod
    def dsigmoid (x):
        sigma = lambda x: 1/(1+exp(-x))
        g = np.vectorize (lambda x: sigma(x)*(1-sigma(x)))
        return g(x)



class InputReader:
    def __init__ (self):
        return 0

class NeuralNet:
    def __init__ (self, inputQuantity, nodesOnLayers): #inputQuantity - int, nodesOnLayers - [int, int, .., int]
        self.nNodes = 0 #quantity of all nodes
        self.nWeights = 0 #quantity of all weights
        self.nLayers = 0 #quantity of layers
        self.lnol = None #list number of nodes on each layers
        self.inp = None # input vector (input nodes)
        self.n = []  # list by layers of nodes vectors
        self.b = []  # list by layers of biases vectors
        self.w = [] # list by layers of weights matrices
        self.z = []  # list by layers of auxiliary vectors of arguments of sigmoid
        #backpropagator vectors
        self.delta = []
        self.dcdw = []
        self.dcdb = [] #unused because self.dcdb = self.delta
        self.layersIndex = []
        if isinstance(nodesOnLayers, list) and isinstance(inputQuantity, int):
            if all(isinstance(ins, int) for ins in nodesOnLayers):
                self.nNodes = inputQuantity
                self.lnol = nodesOnLayers
                self.lnol.insert(0, inputQuantity)
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
                    self.n.append (np.zeros (nodesOnLayers[i], dtype=np.float64)) #initialization of nodes on each layers
                    self.z.append (np.zeros (nodesOnLayers[i], dtype=np.float64))#initialization of arguments of sigmoid on each layers
                    self.b.append (np.random.randn(nodesOnLayers[i]).astype(np.float64)) #initialization of biases on each layers
                    self.w.append(np.random.randn(nodesOnLayers[i], quantityOnPrevious).astype(np.float64)) #initialization of weights on each layers
            else:
                print ("Quantity of nodes must be integer")
        else:
            print ("Please enter neural network construction in following format (numberNodesOnInput, [numberNodesOn1Layer, numberNodesOn2Layer, ... , numberNodesOnNLayer])")

    def backpropagate (self, desOutp): #desOutp - desirable output vector
        y = np.array(desOutp)
        self.delta = []
        self.dcdw = []
        i = self.nLayers-1
        while i>=0:
            if i == (self.nLayers-1):
                self.delta.append((self.output()-y) * AuxiliaryMath.dsigmoid (self.z[i]))
            else:
                self.delta.insert(0, np.transpose(self.w[i+1]).dot(self.delta[0]) * AuxiliaryMath.dsigmoid(self.z[i]))
            i=i-1
        for i in self.layersIndex:
            if i == 0:
                self.dcdw.append (np.outer(self.delta[0], self.inp))
            else:
                self.dcdw.append (np.outer(self.delta[i], self.n[i-1]))



    def input (self, inputParameters):
        if (len(inputParameters) == self.inp.size):
            self.inp = np.array (inputParameters)
        else:
            print ("Please put correct quantity of input values")

    def update (self):
        for i in self.layersIndex:
            if i==0:
                self.z[i] = self.w[i].dot(self.inp)+self.b[i]
                self.n[i] = AuxiliaryMath.sigmoid(self.z[i])
            else:
                self.z[i] = self.w[i].dot(self.n[i-1])+self.b[i]
                self.n[i] = AuxiliaryMath.sigmoid(self.w[i].dot(self.n[i-1])+self.b[i])


    def push (self, eta): # push backpropagation data to parametrs of net
        for i in self.layersIndex:
            self.w[i] = self.w[i]-eta*self.dcdw[i]
            self.b[i] = self.b[i]-eta*self.delta[i]

    def output (self):
        #print (self.n[self.nLayers-1])
        return self.n[self.nLayers-1]

    def verify (self,inputParameters, desOutp):
        self.input(inputParameters)
        self.update ()
        g = np.vectorize (lambda x: pow(x, 2))
        return g(self.output()-desOutp)

    def exportNet (self, filename):
        fo = open (filename, "w")
        #q = range(len(self.lnol))
        for i in self.lnol:
            fo.write(' '+str(i))
        fo.write('\n')
        for matrix in self.w:
            fo.write(":___:")
            for row in matrix:
                fo.write("__")
                for el in row:
                    fo.write(' '+str(el))
        fo.write('\n')
        for vector in self.b:
            fo.write(":_v_:")
            for el in vector:
                fo.write(' '+str(el))
        fo.write('\n')
        fo.close()

    def importNet (self, filename):
        fo = open (filename, 'r')
        specStr = fo.readline().rstrip('\n').lstrip(' ').split(' ')
        wmStrList = fo.readline().rstrip('\n').lstrip(":___:").split(':___:')
        wmRowStrList = [x.lstrip('__').split("__") for x in wmStrList]
        wMatrix = []
        for a in wmRowStrList:
            wMatrix.append([y.lstrip(' ').split(' ') for y in a])
        for nm in range(len(wMatrix)):
            for nr in range(len(wMatrix[nm])):
                for ne in range(len(wMatrix[nm][nr])):
                    self.w[nm][nr][ne]= wMatrix[nm][nr][ne]
        bvStr = fo.readline().rstrip('\n').lstrip(":_v_:").split(':_v_:')
        bVector = [x.lstrip(' ').split(' ') for x in bvStr]

        for nv in range(len(wMatrix)):
            for ne in range(len(wMatrix[nv])):
                self.b[nv][ne] = bVector[nv][ne]
        #print (bVector)
        fo.close()

    def train (self, labels, images, eras, era_len, eta):
        if (len(labels)/era_len<eras):
            print("Too big eras or lenght of era. Please reduce some of them!!!")
        else:
            dcdw_era = []
            delta_era = []
            data = []
            for i in range(len(labels)):
                data.append ([labels[i], images[i]])
            random.shuffle (data)
            '''print (data[2])
            arr = np.zeros ((28, 28), dtype = 'uint8')
            for i in range(28):
                for j in range(28):
                    arr[i][j] = int(data[2][1][i*28+j]*255.0)
            img = Image.fromarray(arr, mode = 'L')
            img.show()'''
            for era in range(eras):
                print('era: ', era)
                dcdw_era = []
                delta_era = []
                for sample in range(era_len):
                    self.input(data[era*era_len+sample][1])
                    self.update()
                    #if (sample==0 and era == 0): print(data[era*era_len+sample][0])
                    self.backpropagate(data[era*era_len+sample][0])
                    #if (sample==0 and era == 0): print(self.delta)
                    dcdw_era.append (self.dcdw)
                    delta_era.append (self.delta)
                    #self.push(eta)

                self.dcdw = list(np.sum(dcdw_era, axis = 0)*(1/era_len))
                self.delta = list(np.sum(delta_era, axis = 0)*(1/era_len))
                self.push(eta)
                #if (era == 0): print (self.delta)
