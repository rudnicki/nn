#!/usr/bin/python

import sys
import math
import random


def sigmoid( total ):
    return 1.0 / ( 1.0 + math.exp(- total) )

class Neuron:
    
    def __init__(self, weights, func=sigmoid):
        self.weights  = weights
        self.func     = func

    def output(self, args):
        # bias self.weights[-1]
        total  = sum( [ self.weights[i] * args[i] for i in range(len(args)) ] )
        total += (-1) * self.weights[-1]
        return self.function(total)
    
    def function(self, arg):
        return self.func(arg)
      
    def setFunction(self, func):
        self.func = func


class Layer:
    #num_inputs
    #output
    def __init__(self):
        self.neurons  = []
		
    def addNeuron(self, neuron):
        self.neurons.append(neuron)

        
class NeuronNetwork:

    def __init__(self):
        self.layers = []
	
        f = open(sys.argv[1], 'r')
        
        [networkInputs, layersNum] = [int(x) for x in f.readline().split()]
        neuronsNums = [networkInputs]
        for i in range(layersNum):
            layerDescription = f.readline().split()
            neuronsNums.append( int(layerDescription[0]) )
            activationFun = layerDescription[1]
			
            L = Layer()
            L.num_inputs = neuronsNums[-2]
            for j in range(neuronsNums[-1]):
                if len(layerDescription) == 4:
                    weights = [ random.uniform(float(layerDescription[2]), float(layerDescription[3])) for i in range(L.num_inputs + 1) ]
                else:
                    weights = [float(x) for x in f.readline().split()]
                L.addNeuron(Neuron(weights, globals()[activationFun]))
            self.addLayer(L)
        f.close()

    def addLayer(self, layer):
        self.layers.append(layer)
		
    def show(self):
        print "Inputs:", self.layers[0].num_inputs
        for idx, layer in enumerate(self.layers):
            print "Layer%d: %d neurons outval =" % (idx, len(layer.neurons)), layer.output   
            for idx, neuron in enumerate(layer.neurons):
                print "\tNeuron%d: weights =" % (idx), neuron.weights
        print "Outputs:", len(self.layers[-1].neurons)
        
    def output(self, inputs):
        if len(inputs) != self.layers[0].num_inputs:
            raise ValueError, 'wrong number of inputs'
        

        for lid, layer in enumerate(self.layers):
            outputs = []
            for n in layer.neurons:
                outputs.append( n.output(inputs) )
            inputs = outputs #for next iteration
            layer.output =  outputs
    
        return self.layers[-1].output

                

print
NN = NeuronNetwork()
print NN.output(
[float(x) for x in sys.argv[2:]])
print
NN.show()
