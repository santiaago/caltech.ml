
from copy import copy

from math import exp
from math import sqrt
from math import log

from numpy import array
from numpy import dot
from numpy import array
from numpy.linalg import norm

from random import shuffle

from tools import target_random_function
from tools import randomline
from tools import data
from tools import build_training_set_fmultipleparams


def err(u,v):
    u = float(u)
    v = float(v)
    return (u*exp(v)-2*v*exp(-u))**2 #non linear error surface

def linear_reg_err(sigma,d,N):
    return (sigma**2)*(1-((d+1)/(N*1.0)))

def deriv_u(u, v):
    u = float(u) 
    v = float(v) 
    return 2*u*exp(2*v) + 4*u*v*exp(v-u) - 4*v*exp(v-u) - 8*v**2*exp(-2*u)

def deriv_v(u, v):
    u = float(u) 
    v = float(v) 
    return 2*u**2*exp(2*v) - 4*u*v*exp(v-u) - 4*u*exp(v-u) + 8*v*exp(-2*u)

def gradient_descent():
    u = 1
    v = 1
    learn_rate = 0.1
    error = 10e-14
    iteration = 0
    
    while(err(u, v) > error):
        delta_u = u - learn_rate*deriv_u(u,v)
        delta_v = v - learn_rate*deriv_v(u,v)
        u,v = delta_u,delta_v
        iteration = iteration + 1
    print "Iteration:%s, u: %s, v: %s E:%s" % (iteration,u,v,err(u,v))

def coordinate_descent():
    u = 1
    v = 1
    learn_rate = 0.1
    error = 10e-14
    iteration = 0
    
    while(err(u, v) > error and iteration < 15):
        u = u - learn_rate*deriv_u(u,v)
        delta_v = v - learn_rate*deriv_v(u,v)
        u,v = u,delta_v
        iteration = iteration + 1
    print "Iteration:%s, u: %s, v: %s E:%s" % (iteration,u,v,err(u,v))

def gradient(sample, y,w):
    vector = [y*1.0] + [y*x*1.0 for x in sample]
    A = array([1]+sample)
    B = array(w)
    div =  (1.0 + exp(y*1.0*dot(A,B)))
    vector = [-1.0 * x / div for x in vector]
    return vector

def log_regression_sgd(t_set,eps,lr):
    w = [0]*len(t_set[0][0])

    converged = False
    nb_epochs = 0
    old_w = []
    index_order = range(len(t_set))
    while not converged:
        old_w = copy(w)
        shuffle(index_order)
        for i in index_order:
            s = t_set[i][0][1:]# get all params expect w0
            y = t_set[i][1]
            grad = gradient(s,y,w)
            
            for j in range(len(w)):
                w[j] = w[j] - lr * grad[j]
        nb_epochs += 1
        W_old = array(old_w)
        W = array(w)
        converged = norm(W_old - W) < eps
    return nb_epochs,w

def log_regression_compute_Eout(t_set,w):
    cee = 0
    for t in t_set:
        x = t[0][1]
        y = t[0][2]
        target = t[1]
        cee += cross_entropy_error([x,y],target,w)
    return cee*1.0/len(t_set)

def cross_entropy_error(sample,y,w):
    A = array([1]+sample)
    B = array(w)
    return log(1.0 + exp(-y*1.0*dot(A,B)))

def run_log_regression():

    nb_in_sample = 100
    nb_out_of_sample = 100000
    nb_runs = 100
    nb_epochs = 0
    nb_Eout = 0
    lr = 0.01
    eps = 0.01

    for i in range(nb_runs):

        l = randomline()
        f = target_random_function(l)
        
        data_in_sample = data(nb_in_sample)
        data_out_of_sample = data(nb_out_of_sample)
        
        t_set_in = build_training_set_fmultipleparams(data_in_sample,f)
        t_set_out = build_training_set_fmultipleparams(data_out_of_sample,f)
        
        epochs,w = log_regression_sgd(t_set_in,eps,lr)
        e_out = log_regression_compute_Eout(t_set_out,w)

        print "Run: %s - epochs: %s"%(i, epochs)
        print "Eout: %s"%(e_out)

        nb_Eout += e_out
        nb_epochs += epochs

    print 'Number of runs:%s'%(nb_runs)
    print "Avg epochs: %s"%(nb_epochs / nb_runs*1.0)
    print "Avg Eout: %s"%(nb_Eout / nb_runs*1.0)

def tests():
    print 'Tests begin'
    print '--------------------'
    print '-1-'
    #sigma = 0.1
    #d = 8
    #print 'sigma:%s d:%s '%(sigma,d)
    #lN = [10,25,100,500,1000]
    #for N in lN:
    #    print 'For N:%s\t Ein:%s '%(N,linear_reg_err(sigma,d,N))
    print '-5-'
    print '-Gradient descent-'
    #gradient_descent()
    print '-7'
    print '-Coordinate descent-'
    #coordinate_descent()
    print '-Logistic regression-'
    print '-8-'
    print '-9-'
    run_log_regression()
