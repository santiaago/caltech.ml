from random import uniform
from random import randint

def data_interval(low_b,high_b,N=100):
    d = []
    for i in range(N):
        d.append(uniform(low_b,high_b))
    return d
def data(N = 10):
    'return N random points (x1,x2)'
    d = []
    for i in range(N):
        x =  uniform(-1,1)
        y =  uniform(-1,1)
        d.append([x,y])
    return d

def randomline():
    'computes a random line and returns a and b params: y = ax + b'
    x1 = uniform(-1,1)
    y1 = uniform(-1,1)
    x2 = uniform(-1,1)
    y2 = uniform(-1,1)
    
    a = abs(x1-x2)/abs(y1-y2)
    b = y1 - a*x1
    return [a,b] # a*x + b

def target_function(l):
    # print 'Target function: %s x + %s' %(l[0], l[1]
    f = lambda x: l[0]*x + l[1]
    return f

def sign(x,compare_to = 0):
    'returns +1 or -1 by comparing x to compare_to param (by default = 0)'
    if x > compare_to:
        return +1.
    else:
        return -1.

def map_point(point,f):
    'maps a point (x1,x2) to a sign -+1 following function f '
    x1 = point[0]
    y1 = point[1]
    
    y = f(x1)
    compare_to = y1
    return sign(y,compare_to)

def map_point_fmultipleparams(point,f):
    y1 = point[1]
    y = f(point)
    compare_to = y1
    return sign(y,compare_to)

def build_training_set(data, func):
    t_set = []
    for i in range(len(data)):
        point = data[i]
        y = map_point(point,func)
        t_set.append([ [ 1.0, point[0],point[1] ] , y ])
    return t_set

def build_training_set_fmultipleparams(data,func):
    t_set = []
    for i in range(len(data)):
        point = data[i]
        y = map_point_fmultipleparams(point,func)
        t_set.append([ [ 1.0, point[0],point[1] ] , y ])
    return t_set


def print_avg(name,vector):
    print 'Average %s: %s'%(name,sum(vector)/(len(vector)*1.0))
