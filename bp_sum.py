#!/usr/bin/python

from nn import *

pattern = [[0, 1, 1, 
            0, 1, 1, 
            0, 1, 1],
           
           [0, 1, 0,
            0, 1, 0, 
            0, 1, 0],
           
           [0, 0, 1, 
            0, 0, 1, 
            0, 0, 1]
           ] 

classes = [[1, 0 ,0],
           [0, 1, 0],
           [0, 0, 1]
          ]

NN = BackPropagationNetwork(sys.argv[1], M=0.2, N=0.5, with_bias=True)

print "\n<<Initial weights>>"
NN.show()

#learning the pattern
NN.learnBP(pattern, classes, iterations = 1000)

print "\n<<Weights after learning>>"
NN.show()

#test pattern vectors
print "\n<<Validate>>"
print "input_vector", "---->", "output_vector", "---->", "winner id"

for x in pattern:
    output = NN.output(x)
    max_idx, max_val = max(enumerate(output), key=operator.itemgetter(1))
    print short(x), "---->", short(NN.out()), "---->", max_idx
