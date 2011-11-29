#!/usr/bin/python

from nn import *

NN = NeuronNetwork(sys.argv[1])
inputs = [float(x) for x in sys.argv[2:]]
print NN.output(inputs)
