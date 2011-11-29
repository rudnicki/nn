#!/usr/bin/python

import sys
import math
import random
import operator

def sigmoid( total ):
    return 1.0 / ( 1.0 + math.exp(- total) )

def normalize(vec):
    scale = math.sqrt( sum( [v*v for v in vec] ))
    return [ v / scale for v in vec ]

def dist(u, v):
    sub2 = [ pow(ui - vi, 2) for ui, vi in zip(u, v) ]
    return math.sqrt( sum( sub2 ))

class Neuron:
    
    def __init__(self, weights, func=sigmoid):
        self.weights  = weights
        self.func     = func
        #kohonen
        self.pmin = 0 #0.75
        self.p = 1

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
    #self.num_inputs
    #self.output
    def __init__(self):
        self.neurons  = []
		
    def addNeuron(self, neuron):
        self.neurons.append(neuron)

class KohonenLayer(Layer):
    def __init__(self):
        #super(KohonenLayer, self).__init__()
        Layer.__init__(self)
        self.eta = 0.1
        
    def winner(self, x):
        dists = [ dist(n.weights, x) for n in self.neurons if n.p > n.pmin]
        min_idx, min_val = min(enumerate(dists), key=operator.itemgetter(1))
        return min_idx
    
    def update_p(self,win_idx):
        for n_idx, n in enumerate(self.neurons):
            if n_idx == win_idx:
                n.p = n.p - n.pmin
            else:
                n.p = n.p + 1.0/len(self.neurons) #TODO what is n in 1/n for pi ?
        
    def learn_step(self, x):
        k_idx = self.winner(x)
        k_neuron = self.neurons[ k_idx ] # winner neuron
        
        #update winner weights
        k_neuron.weights = normalize( [ wi + self.eta * (xi - wi) for xi, wi in zip(x, k_neuron.weights) ] )
        self.update_p(k_idx)


class NeuronNetwork():
    def __init__(self, kohonen = False):
        self.layers = []
	self.kohonen = kohonen

        f = open(sys.argv[1], 'r')
        
        [networkInputs, layersNum] = [int(x) for x in f.readline().split()]
        neuronsNums = [networkInputs]
        for i in range(layersNum):
            layerDescription = f.readline().split()
            neuronsNums.append( int(layerDescription[0]) )
            activationFun = layerDescription[1]
		
            if(self.kohonen):
                L = KohonenLayer()
            else:
                L = Layer()
            L.num_inputs = neuronsNums[-2]
            for j in range(neuronsNums[-1]):
                if len(layerDescription) == 4:
                    weights = [ random.uniform(float(layerDescription[2]), float(layerDescription[3])) 
                                for i in range(L.num_inputs) ]
                    #bias
                    if(self.kohonen):
                        weights.append( 0.0 )
                    else:
                        weights.append( random.uniform(float(layerDescription[2]), float(layerDescription[3]))  )
                else:
                    weights = [float(x) for x in f.readline().split()]
                weigths = normalize(weights)
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

    def out(self):
        return self.layers[-1].output
        
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



pattern1 = map(normalize, [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]] )

pattern2 = map(normalize, [[1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 0, 0, 1],
                           [1, 1, 1, 1]] ) 

                
# create NeutralNetwork and pass input vector
NN = NeuronNetwork(kohonen = True)
NN.output([float(x) for x in sys.argv[2:]])

print "\n<<Initial weights>>"
NN.show()

#shuffle pattern for learn
pattern = pattern1 
learnset = [ pattern[random.randint(0,len(pattern)-1)] for i in range(23) ]

for i in range(1000):
    for x in learnset:        
        NN.output(x)
        NN.layers[-1].learn_step(x)


print "\n<<Weights after learning>>"
NN.show()

#test pattern vectors        
print "\n<<Validate>>"
print "input_vector", "---->", "output_vector", "---->", "winner id"
for x in pattern:
    NN.output(x)
    max_idx, max_val = max(enumerate(NN.out()), key=operator.itemgetter(1))
    print x, "---->", NN.out(), "---->", max_idx


