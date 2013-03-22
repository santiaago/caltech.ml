from numpy import array
from numpy import dot
from numpy import sign
from numpy import transpose
from numpy import identity
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import pinv # pseudo inverse aka dagger

from tools import data_from_file
from tools import input_data_matrix
from tools import linear_regression
from tools import pseudo_inverse
from tools import target_vector

KA = -3
LAMBDA = 10**KA

def compute_Ein(wlin, X, y):
    'fraction of in sample points which got classified incorrectly'
    N = len(y)
    g_vector = sign(dot(X,wlin))
    
    vEin = g_vector - y
    nEin = 0
    for i in range(len(vEin)):
        if vEin[i]!= 0: nEin = nEin + 1

    return nEin / (len(vEin) *1.0) 

def compute_Eout_nonlineartrans(wlin,outdata):
    N_points = len(outdata)

    t_set_trans = transform_t_set(outdata)
    X_matrix = input_data_matrix(t_set_trans)
    y_vector = target_vector(t_set_trans)
    
    g_vector = dot(X_matrix,wlin)
    for i in range(len(g_vector)):
        g_vector[i] = sign(g_vector[i])
    vEout = g_vector - y_vector
    nEout = 0
    for i in range(len(vEout)):
        if vEout[i] != 0:
            nEout = nEout+1
    Eout = nEout / (len(vEout)*1.0)
    return Eout


def transform_t_set(data):
    trans_data=[]

    for i in range(len(data)):
        x1 = data[i][0][0]
        x2 = data[i][0][1]
        y = data[i][1]
        tX = [1,x1,x2,x1**2,x2**2,x1*x2,abs(x1-x2),abs(x1+x2)]
        trans_data.append([ tX,y ])
    return trans_data

def compute_weight_decay(wlin,t_set,X,y,k):
    sumsq = 0
    for wi in wlin:
        sumsq = sumsq + wi**2
    wdc = ((10**k)/(len(t_set)*1.0))*sumsq
    t1 = dot(transpose(X),X)
    t2 = t1 + wdc*identity(len(t1))
    t3 = inv(t2)
    t4 = dot(t3,transpose(X))
    w_decay = dot(t4,y)
    return w_decay

def run_nonlineartransformation(indata,outdata):
    N_points = len(indata)

    t_set_trans = transform_t_set(indata)
    wtrans,Xtrans,ytrans = linear_regression(N_points,t_set_trans)
    print '-2-'
    print 'Linear regression on training set after non linear transformation:'
    Eintrans = compute_Ein(wtrans,Xtrans,ytrans)
    Eouttrans = compute_Eout_nonlineartrans(wtrans,outdata)
    print 'in sample classification error: %s'%(Eintrans)
    print 'out of sample classification error: %s'%(Eouttrans)
    print '-3-'
    print 'Adding weight decay to linear regression with lambda = 10k and k = -3'
    w_decay = compute_weight_decay(wtrans,t_set_trans,Xtrans,ytrans,-3)
    Eintrans_decay = compute_Ein(w_decay,Xtrans,ytrans)
    Eouttrans_decay=compute_Eout_nonlineartrans(w_decay,outdata)
    print 'in sample classification error:%s'%(Eintrans_decay)
    print 'out of sample classification error: %s'%(Eouttrans_decay)
    print '-4-'
    print 'Using now k = 3'
    w_decay = compute_weight_decay(wtrans,t_set_trans,Xtrans,ytrans,3)
    Eintrans_decay = compute_Ein(w_decay,Xtrans,ytrans)
    Eouttrans_decay=compute_Eout_nonlineartrans(w_decay,outdata)
    print 'in sample classification error: %s'%(Eintrans_decay)
    print 'out of sample classification error: %s'%(Eouttrans_decay)
    print '-5-'
    Ks = [2,1,0,-1,-2]
    print 'searching the lowest out of sample classification error for the following k values.'
    print 'k in (%s)'%(str(Ks))
    for k in Ks:
            w_decay = compute_weight_decay(wtrans,t_set_trans,Xtrans,ytrans,k)
            Eintrans_decay = compute_Ein(w_decay,Xtrans,ytrans)
            Eouttrans_decay=compute_Eout_nonlineartrans(w_decay,outdata)
            print 'K : %s'%(k)
            print 'in sample classification error: %s'%(Eintrans_decay)
            print 'out of sample classification error: %s'%(Eouttrans_decay)
    print '-6-'
    print 'searching the minimum out of sample classification error by varying k in the integer values.'
    mink = 999
    minEout = 999
    for k in range(-200,200):
        w_decay = compute_weight_decay(wtrans,t_set_trans,Xtrans,ytrans,k)
        Eintrans_decay = compute_Ein(w_decay,Xtrans,ytrans)
        Eout_decay=compute_Eout_nonlineartrans(w_decay,outdata)
        if Eout_decay < minEout:
            minEout = Eout_decay
            mink = k
    print 'K: %s'%(k)
    print 'out of sample classification error: %s'%(minEout)

def tests():
    print '-1-'
    indata  = data_from_file('in.dta')
    outdata = data_from_file('out.dta')
    run_nonlineartransformation(indata,outdata)
    print '-8-'
    print '-9-'
    print '-10-'
