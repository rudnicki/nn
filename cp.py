#!/usr/bin/python

from nn import *

pattern = [[1, 1, 1, 
            0, 0, 0, 
            0, 0, 0],
           [0, 0, 0,
            1, 1, 1, 
            0, 0, 0],   
           [0, 0, 0,
            0, 0, 0, 
            1, 1, 1], 	
           [1, 0, 0,
            1, 0, 0, 
            1, 0, 0],
           [0, 1, 0,
            0, 1, 0, 
            0, 1, 0],			
           [0, 0, 1,
            0, 0, 1, 
            0, 0, 1],
		   [1, 0, 0,
            0, 0, 0, 
            0, 0, 0],
           [0, 0, 0,
            0, 0, 0, 
            0, 0, 1],
           [0, 0, 1,
            0, 1, 0, 
            1, 0, 0],			
           ] 

classes = [[1, 0, 0],
		   [1, 0, 0],
		   [1, 0, 0],
		   [0, 1, 0],
		   [0, 1, 0],
		   [0, 1, 0],
		   [0, 0, 1],
		   [0, 0, 1],
		   [0, 0, 1]]
		   
NN = CounterPropagationNetwork(sys.argv[1])

print "\n<<Initial weights>>"
NN.show()

#learning the pattern
iterations = 10000
epochEtas = [0.8, 0.6, 0.4, 0.1]
epochNeig = [0, 0, 0, 0]
iterationsPerEpoch = 4000
roMin = 0.8
neighbourhoodDim = 1
alfa = 0.05
NN.learnCP(iterations, classes, pattern, alfa, epochEtas, iterationsPerEpoch, roMin, epochNeig, neighbourhoodDim)

print "\n<<Weights after learning>>"
NN.show()

#test pattern vectors
print "\n<<Validate>>"
print "input_vector", "---->", "output_vector", "---->", "winner id"

for x in pattern:
    NN.output(x, True)
    max_idx, max_val = NN.find_winner()
    print short(x), "---->", short(NN.out()), "---->", max_idx
