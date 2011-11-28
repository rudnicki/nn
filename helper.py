import math

def normalize(vec):
    scale = math.sqrt( sum( [v*v for v in vec] ))
    return [ v / scale for v in vec ]

def dist(u, v):
    sub2 = [ pow(ui - vi, 2) for ui, vi in zip(u, v) ]
    return math.sqrt( sum( sub2 ))


vec = [3, 1, 2]
w = [1, 2, 0]
v = [-2, 3, 5]
print dist(w, v)
print vec, normalize(vec)
