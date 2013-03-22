from math import log
from math import sin
from math import sqrt
from math import pi

from random import uniform

from tools import data_interval

def generalization_error(dvc,confidence,gen_err, iterations=10):
    'from the VC generalization bound and using the Sample Complexity algorithm page:57'
    N = 1000 #start with an initial sample of 1000
    for i in range(iterations):
        v = (4*((N*2)**dvc)+4)/confidence
        N = (8/(gen_err**2))*log(v)
    return N

def VC_bound(N,dvc,confidence):
    e = sqrt((8/(N*1.0))*log((4*((N*2)**dvc)+4)/confidence))
    return e
def Rademacher_Penalty_Bound(N,dvc,confidence):
    e = sqrt((2*log(2*N*(N**dvc)+2*N)/(N*1.0)))+ sqrt((2/(N*1.0))*log(1/confidence)) + 1/(N*1.0)
    return e
def ParrondoandVandenBroek(N,dvc,confidence):
    e = (sqrt((1+N**2)+log((6*((2*N+1)**dvc)+6)/confidence))+1)/(N*1.0)
    return e
def Devroye(N,dvc,confidence):
    #use 2**N instead of dvc as it is a bound
    e = ((sqrt((N*log((4*((2**N)))/confidence)+(1/(N-2)*1.0)*(N-2))) +1)/((N-2)*1.0))
    return e
def pick_point(f):
    'returns x coords randomly from interval -1 to 1 and y from function f'
    x = uniform(-1,1)
    y = f(x)
    return x,y
def mean_sqrt_error(x1,y1,x2,y2):
    return abs(x2-x1)/2.,abs(y2-y1)/2.

def print_info_bias_variance(slope,constant,bias,var):
    if slope is not None: print 'a: %s'%slope
    if constant is not None: print 'b: %s'%constant
    print 'bias: %s'%bias
    print 'var: %s'%var
    print 'Eout: %s'%(bias+var)

def compute_bias(bias_fn,f):
    table_bias = []
    for i in range(100):
        x = uniform(-1,1)
        gx = bias_fn(x)
        fx = f(x)
        table_bias.append((gx-fx)**2)
    bias = sum(table_bias)/(len(table_bias)*1.0)
    return bias

def compute_var(var_fn, table_g):
    table_var = []
    for i in range(100):
        x = uniform(-1,1)
        gbarx = var_fn(x)
        table_one_g = []
        for g in table_g:
            gx = g(x)
            table_one_g.append((gx - gbarx)**2)
        table_var.append(sum(table_one_g)/(len(table_one_g)*1.0))
    var = sum(table_var)/(len(table_var)*1.0)
    return var
def compute_var_ex(var_fn,table_a,table_b):
     table_var = []
     for i in range(100):
         x = uniform(-1,1)
         gbarx = var_fn(x)
         table_one_g = []
         for i in range(len(table_a)):
             gx = table_a[i]*x**2 + table_b[i]
             table_one_g.append((gx - gbarx)**2)
         table_var.append(sum(table_one_g)/(len(table_one_g)*1.0))
     var = sum(table_var)/(len(table_var)*1.0)
     return var
def bias_and_variance():
    f = lambda x:sin(x*pi)
    table_g = []
    table_a = []
    for i in range(100):
        x1,y1 = pick_point(f)
        x2,y2 = pick_point(f)
        x,y = mean_sqrt_error(x1,y1,x2,y2)
        a = y/x #slope
        #compute g
        g1 = lambda x:a*x
        table_g.append(g1)
        table_a.append(a)
    slope = sum(table_a)/(len(table_a)*1.0)
    #bias
    bias_fn = lambda x: slope*x
    bias = compute_bias(bias_fn,f)
    #variance
    var_fn = lambda x: slope*x
    var = compute_var(var_fn,table_g)
    #print info
    print_info_bias_variance(slope,None,bias,var)

def bias_and_variance_constant():
    f = lambda x:sin(x*pi)
    table_g = []
    table_b = []
    for i in range(100):
        x1,y1 = pick_point(f)
        x2,y2 = pick_point(f)
        x,y = mean_sqrt_error(x1,y1,x2,y2)
        b = y
        #compute g
        g1 = lambda x:b
        table_g.append(g1)
        table_b.append(b)
    constant = sum(table_b)/(len(table_b)*1.0)
    #bias
    bias_fn = lambda x: constant
    bias = compute_bias(bias_fn,f)
    #variance
    var_fn = lambda x: constant
    var = compute_var(var_fn,table_g)
    #print info
    print_info_bias_variance(None,constant,bias,var)

def bias_and_variance_function():
     f = lambda x:sin(x*pi)
     table_g = []
     table_a = []
     table_b = []
     for i in range(100):
         x1,y1 = pick_point(f)
         x2,y2 = pick_point(f)
        #slope
         a = (y1-y2)/(x1-x2)
         b = y1 - a*x1
        #compute g
         table_a.append(a)
         table_b.append(b)
     slope = sum(table_a)/(len(table_a)*1.0)
     constant = sum(table_b)/(len(table_b)*1.0)
    #bias
     bias_fn = lambda x: slope*x+constant
     bias = compute_bias(bias_fn,f)
    #variance
     var_fn = lambda x: slope*x
     var = compute_var_ex(var_fn,table_a,table_b)
     #print info
     print_info_bias_variance(None,constant,bias,var)

def bias_and_variance_square():
    f = lambda x:sin(x*pi)
    table_g = []
    table_a = []
    for i in range(100):
        x1,y1 = pick_point(f)
        x2,y2 = pick_point(f)
        x,y = mean_sqrt_error(x1,y1,x2,y2)
        a = y/x #slope
        #compute g
        g1 = lambda x:a*x**2
        table_g.append(g1)
        table_a.append(a)
    slope = sum(table_a)/(len(table_a)*1.0)
    #bias
    bias_fn = lambda x: slope*x**2
    bias = compute_bias(bias_fn,f)
    #variance
    var_fn = lambda x: slope*x**2
    var = compute_var(var_fn,table_g)
    #print info
    print_info_bias_variance(slope,None,bias,var)

def bias_and_variance_square_constant():
     f = lambda x:sin(x*pi)
     table_g = []
     table_a = []
     table_b = []
     for i in range(100):
         x1,y1 = pick_point(f)
         x2,y2 = pick_point(f)
        #slope
         a = (y1-y2)/(x1-x2)
         b = y1 - a*x1**2
        #compute g
         table_a.append(a)
         table_b.append(b)
     slope = sum(table_a)/(len(table_a)*1.0)
     constant = sum(table_b)/(len(table_b)*1.0)
    #bias
     bias_fn = lambda x: slope*x**2 + constant
     bias = compute_bias(bias_fn,f)
    #variance
     var_fn = lambda x: slope*x**2
     var = compute_var_ex(var_fn,table_a,table_b)
     #print info
     print_info_bias_variance(None,constant,bias,var)

def tests():
    print 'Tests begin'
    print '--------------------'
    print '-1-'
    #1
    dvc = 10
    confidence = 0.05
    gen_err = 0.05
    N = generalization_error(dvc,confidence,gen_err)
    print 'Sample Size for dvc:%s with confidence of %s and with generalization error of %s is %s'%(dvc,confidence,gen_err,N)
    #2
    print '-2-'
    dvc = 50
    confidence = 0.05
    N = 1000
    print 'experience with dvc=%s, confidence of %s and %s samples'%(dvc,confidence,N)
    print 'Original VC bound: %s'%(VC_bound(N,dvc,confidence))   
    print 'Rademacher Penalty bound: %s'%(Rademacher_Penalty_Bound(N,dvc,confidence))
    print 'Parrondo and Vanden Broek bound: %s'%(ParrondoandVandenBroek(N,dvc,confidence))
    print 'Devroye bound: %s' %(Devroye(N,dvc,confidence))
    #3
    print '_3_'
    N = 5
    print 'experience with dvc=%s, confidence of %s and %s samples'%(dvc,confidence,N)
    print 'Original VC bound: %s'%(VC_bound(N,dvc,confidence))   
    print 'Rademacher Penalty bound: %s'%(Rademacher_Penalty_Bound(N,dvc,confidence))
    print 'Parrondo and Vanden Broek bound: %s'%(ParrondoandVandenBroek(N,dvc,confidence))
    print 'Devroye bound: %s' %(Devroye(N,dvc,confidence))
    #4
    print '-4-5-6-'
    bias_and_variance()
    print '-7-'
    bias_and_variance_constant()
    print '--'
    bias_and_variance_function()
    print '--'
    bias_and_variance_square()
    print '--'
    bias_and_variance_square_constant()
    print '--------------------'
    print 'Tests end'
