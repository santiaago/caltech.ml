from random import uniform
from random import randint

from numpy import array
from numpy.linalg import pinv as pinv # pseudo inverse aka dagger
from numpy import dot
from numpy import eye
from numpy import size

def data_interval(low_b,high_b,N=100):
    'returns a list of N values uniformly distributed between low boundary and high boundary'
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

def data_from_file(filepath):
    'from a filepath returns a dataset with the form [[x1,x2],y]'
    datafile = open(filepath, 'r')
    data = []
    for line in datafile:
        split = line.split()
        x1 = float(split[0])
        x2 = float(split[1])
        y = float(split[2])
        data.append([ [x1,x2],y ])
    return data

def randomline():
    'computes a random line and returns [a,b] : y = ax + b'
    x1 = uniform(-1,1)
    y1 = uniform(-1,1)
    x2 = uniform(-1,1)
    y2 = uniform(-1,1)
    
    a = abs(x1-x2)/abs(y1-y2)
    b = y1 - a*x1
    return [a,b] # a*x + b

def target_function(coords):
    'from a coordinate input [a,b] returns the function a*x + b'
    f = lambda x: coords[0]*x + coords[1]
    return f

def target_random_function(coords):
    '''
    description: from a coordinate (coords) with the format [a,b] generated a random function.
    - coord: a list of the form [a,b]
    - returns: the generated random function that takes as argument a list with the form [x,y]
    and returns 1 or -1 whether y is below the linear function defined by a*x + b or above.
    '''
    func = target_function(coords)
    def f(X):
        x = X[0]
        y = X[1]
        if func(x) < y:
            return 1.0
        else:
            return -1.0
    return f

def signex(x,compare_to = 0):
    'returns +1 or -1 by comparing (x) to (compare_to) param (by default = 0)'
    if x > compare_to:
        return +1.
    else:
        return -1.

def sign(x,compare_to = 0):
    'returns +1 or -1 by comparing (x) to (compare_to) param (by default = 0)'
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

def target_vector(t_set):
    'creates a numpy array (eg a Y matrix) from the training set'
    y = array([t[1] for t in t_set])
    return y

def input_data_matrix(t_set):
    'creates a numpy array (eg a X matrix) from the training set'
    X = array([t[0] for t in t_set])
    return X

def pseudo_inverse(X):
    'dagger of pseudo matrix used for linear regression'
    return pinv(X)

def linear_regression(N_points,t_set):
    '''Linear regresion algorithm
    from Y and X compute the dagger or pseudo matrix
    return the Xdagger.Y as the w vector
    default lambda is 1.0
    '''
    y_vector = target_vector(t_set)
    X_matrix = input_data_matrix(t_set)
    X_pseudo_inverse = pseudo_inverse(X_matrix)
    return dot(dot(X_pseudo_inverse,X_matrix.T),y_vector),X_matrix,y_vector

def linear_regression_lda(N_points,t_set,lda):
    '''Linear regresion algorithm
    from Y and X compute the dagger or pseudo matrix
    return the Xdagger.Y as the w vector
    default lambda is 1.0
    '''
    y_vector = target_vector(t_set)
    X_matrix = input_data_matrix(t_set)
    X_pseudo_inverse = pseudo_inverse(dot(X_matrix.T,X_matrix)+lda*eye(size(X_matrix,1)))
    return dot(dot(X_pseudo_inverse,X_matrix.T),y_vector),X_matrix,y_vector

