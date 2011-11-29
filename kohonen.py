#!/usr/bin/python

from nn import *

pattern = [[0, 0, 1, 
            0, 0, 1, 
			0, 0, 1],
						   
           [1, 0, 0, 
			0, 1, 0, 
			0, 0, 1],
						   
           [1, 1, 1, 
			1, 0, 1, 
			1, 1, 1],
						   
           [0, 1, 0,
			1, 0, 1, 
			0, 1, 0]
		] 

NN = NeuronNetwork(sys.argv[1], kohonen = True)

print "\n<<Initial weights>>"
NN.show()

#learning the pattern
epochEtas = [0.8, 0.6, 0.4, 0.1]
iterationsPerEpoch = 2000
roMin = 0.3
neighbourhood = 0
neighbourhoodDim = 1
NN.learn(pattern, epochEtas, iterationsPerEpoch, roMin, neighbourhood, neighbourhoodDim)

print "\n<<Weights after learning>>"
NN.show()

#NN.save("kohonen_nauczony.txt")

#test pattern vectors
print "\n<<Validate>>"
print "input_vector", "---->", "output_vector", "---->", "winner id"

for x in pattern:
    NN.output(x, True)
    max_idx, max_val = max(enumerate(NN.out()), key=operator.itemgetter(1))
    print short(x), "---->", short(NN.out()), "---->", max_idx