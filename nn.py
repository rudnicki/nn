#!/usr/bin/python

import sys
import math
import random
import operator

def linear( total ):
    return total

def linear_derivative( total ):
    return 1.0
	
def sigmoid( total ):
    return 1.0 / ( 1.0 + math.exp(- total) )

def sigmoid_derivative( total ):
    return sigmoid(total) * (1.0 - sigmoid(total))
	
def normalize(vec):
    scale = math.sqrt( sum( [v*v for v in vec] ))
    if scale==0:
        return vec
    else:
        return [ v / scale for v in vec ]

def euk_dist(u, v):
    sub2 = [ pow(ui - vi, 2) for ui, vi in zip(u, v) ]
    return math.sqrt( sum( sub2 ))

def short(vec):
    nums = ' '.join(["%.2f" % (vi) for vi in vec ])
    return "[" + nums + "]"
    
class Neuron:
    
    def __init__(self, weights, func='sigmoid', bweight = 0):
        self.weights = weights
        self.bweight = bweight
        self.func = globals()[func]
        self.derivative = globals()[func + '_derivative']
        #kohonen
        self.ro = 1

    def output(self, args):
        total = sum( [ self.weights[i] * args[i] for i in range(len(args)) ] )
        total += (-1) * self.bweight
        return self.function(total)
    
    def function(self, arg):
        return self.func(arg)
      

class Layer:
    def __init__(self):
        self.neurons = []
        self.num_inputs = 0
        self.output = []

    def addNeuron(self, neuron):
        self.neurons.append(neuron)

class KohonenLayer(Layer):
    def __init__(self, dim=1):
        Layer.__init__(self)
        self.eta  = 0.1
        self.neig = 0
        self.dim  = dim
        self.romin = 0.5

    def size(self):
        if(self.dim == 1):
            return len(self.neurons)
        if(self.dim == 2):
            return math.sqrt(len(self.neurons))
    ssize = property(size) # ssize - side_size

    def winner(self, x):
        dists = []
        for n in self.neurons:
            if n.ro > self.romin:
                dists.append( euk_dist(n.weights, x) )
            else:
                dists.append( float("inf") )
        
        min_idx, min_val = min(enumerate(dists), key=operator.itemgetter(1))
        
        return min_idx

    def is_neig(self, win_id, other_id):
        if win_id == other_id:
            return 1
        elif abs(win_id - other_id) <= self.neig:
            return 0.5
        # 2dimension case
        elif (win_id % self.ssize == other_id % self.ssize) and ( abs(win_id - other_id) <= self.ssize * self.neig):
            return 0.5
        else:
            return 0

    def update_ro(self, win_idx):
        for n_idx, n in enumerate(self.neurons):
            if n_idx == win_idx:
                n.ro = n.ro - self.romin
            else:
                n.ro = min( n.ro + 1.0/len(self.neurons), 1)
        
    def learn_step(self, x):
        k_idx = self.winner(x)
        k_neuron = self.neurons[ k_idx ] # winner neuron
        
        #update winner and neighbours
        for n_id, n in enumerate(self.neurons):
            new_weights = [ wi + self.is_neig(k_idx, n_id) * self.eta * (xi - wi) for xi, wi in zip(x, n.weights) ]
            n.weights   = normalize( new_weights )
        self.update_ro(k_idx)
		
class NeuronNetwork():
    def __init__(self, filename, kohonen=False):
        self.layers = []
        self.kohonen = kohonen

        f = open(filename, 'r')
        
        [networkInputs, layersNum] = [int(x) for x in f.readline().split()]
        neuronsNums = [networkInputs]
        for i in range(layersNum):
            layerDescription = f.readline().split()
            neuronsNums.append( int(layerDescription[0]) )
            activationFun = layerDescription[1]

            L = self.createLayer(i);
			
            L.num_inputs = neuronsNums[-2]
            for j in range(neuronsNums[-1]):
                if len(layerDescription) == 4:
                    weights = [ random.uniform(float(layerDescription[2]), float(layerDescription[3]))
                                for i in range(L.num_inputs) ]
                    #weights = [0.0 for i in range(L.num_inputs)]
                    #bias
                    if(self.kohonen):
                        bweight = 0.0
                        weigths = normalize(weights)
                    else:
                        bweigth = random.uniform(float(layerDescription[2]), float(layerDescription[3]))
                else:
                    weights = [float(x) for x in f.readline().split()]
                    bweight = weights.pop()
                L.addNeuron(Neuron(weights, activationFun, bweight))
            self.addLayer(L)
        f.close()

	def createLayer(self):
		return Layer();
		
    def addLayer(self, layer):
        self.layers.append(layer)

    def show(self):
        print "Inputs:", self.layers[0].num_inputs
        for idx, layer in enumerate(self.layers):
            print "Layer%d: %d neurons outval =" % (idx, len(layer.neurons)), short(layer.output)
            for idx, neuron in enumerate(layer.neurons):
                print "\tNeuron%d: weights =" % (idx), short(neuron.weights), "bias_weight=",neuron.bweight
        print "Outputs:", len(self.layers[-1].neurons)

    def out(self):
        return self.layers[-1].output
        
    def output(self, inputs, normalizeInputs=False):
        if len(inputs) != self.layers[0].num_inputs:
            raise ValueError, 'wrong number of inputs'
        
        if(normalizeInputs == True):
            inputs = map(normalize, [inputs])[0]
        for lid, layer in enumerate(self.layers):
            outputs = []
            for n in layer.neurons:
                outputs.append( n.output(inputs) )
            inputs = outputs #for next iteration
            layer.output = outputs
    
        return self.layers[-1].output

    def save(self, filename):
        f = open(filename, 'w')
        f.write(str(len(self.layers[0].neurons[0].weights)) + " " + str(len(self.layers)))
        f.write("\n")
        for layer in self.layers:
            f.write(str(len(layer.neurons)) + " " + layer.neurons[0].func.__name__)
            f.write("\n")
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    f.write(str(weight) + " ")
                f.write(str(neuron.bweight))
                f.write("\n")
            f.write("\n")
        f.close()
		
class KohonenNetwork(NeuronNetwork):
    def __init__(self, filename):
        NeuronNetwork.__init__(self, filename, True)
		
    def createLayer(self, i):
        return KohonenLayer();
	
    def getKohonenLayer(self):
        return self.layers[-1]
	
    def learn(self, pattern, epochEtas, iterationsPerEpoch, romin, epochNeigs, dim):
        kohonenLayer = self.getKohonenLayer()
        kohonenLayer.dim = dim
        kohonenLayer.romin = romin

        pattern = map(normalize, pattern)
        for epochEta, epochNeig in zip(epochEtas, epochNeigs):
            for i in range(iterationsPerEpoch):
                kohonenLayer.eta = epochEta
                kohonenLayer.neig = epochNeig
                #x = pattern[random.randint(0,len(pattern)-1)]
                x = pattern[i % len(pattern)]
                self.output(x)
                kohonenLayer.learn_step(x)
				
    def find_winner(self):
        return max(enumerate(self.layers[-1].output), key=operator.itemgetter(1))

class GrossbergLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

    def learn_step(self, input, actual_output, target_output, alfa):
        for i in range(0, len(self.neurons)):
            n = self.neurons[i]
            diff = target_output[i] - actual_output[i]
            weighted_sum = sum([inp * w for (inp, w) in zip(input, n.weights)])
            new_weights = [w + alfa*(x * diff)*n.derivative(weighted_sum) for (x, w) in zip(input, n.weights)]
            n.weights = new_weights

class CounterPropagationNetwork(KohonenNetwork):
    def __init__(self, filename):
        NeuronNetwork.__init__(self, filename, True)
        self.alfa = 0.3
		
    def createLayer(self, i):
        if i == 0:
            return KohonenLayer()
        else:
            return GrossbergLayer()
			
    def getKohonenLayer(self):
        return self.layers[-2]
	
    def getGrossbergLayer(self):
        return self.layers[-1]
	
    def learnCP(self, iterations, classes, pattern, alfa, epochEtas, iterationsPerEpoch, romin, epochNeigs, dim):
        self.alfa = alfa
        
	    #naucz warstwe Kohonena
        self.learn(pattern, epochEtas, iterationsPerEpoch, romin, epochNeigs, dim)
		
        #naucz warstwe Grossberga
        for i in range(0, iterations):
            for p in range(0, len(pattern)):
                pattern_idx = random.randint(0,len(pattern)-1)
                x = pattern[pattern_idx]
                target_output = classes[pattern_idx]
                actual_output = self.output(x)
                input = self.getKohonenLayer().output
                self.getGrossbergLayer().learn_step(input, actual_output, target_output, self.alfa)
	
    def output(self, inputs):
        if len(inputs) != self.layers[0].num_inputs:
            raise ValueError, 'wrong number of inputs'
        
        inputs = map(normalize, [inputs])[0]
        outputs = []
        for n in self.getKohonenLayer().neurons:
            outputs.append( n.output(inputs) )
		
        #w outpucie warstwy kohonena ustawiamy winnera na 1.0, reszta na 0.0		
        winner_idx, winner = max(enumerate(outputs), key=operator.itemgetter(1))
        outputs = [0.0] * len(outputs)
        outputs[winner_idx] = 1.0		
        self.getKohonenLayer().output = outputs
		
        inputs = self.getKohonenLayer().output
        outputs = []
        for n in self.getGrossbergLayer().neurons:
            outputs.append( n.output(inputs) )		
        self.getGrossbergLayer().output = outputs;
		
        return self.layers[-1].output