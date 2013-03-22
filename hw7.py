#cvxopt Python Software for Convex Optimization: http://abel.ee.ucla.edu/cvxopt/
from cvxopt import solvers
from cvxopt import spmatrix
from cvxopt import matrix

from numpy import array
from numpy import dot
from numpy import sign

from math import fabs
from math import sqrt

from hw1 import evaluate_diff_f_g
from hw1 import PLA

from tools import build_training_set
from tools import data
from tools import data_from_file
from tools import input_data_matrix
from tools import linear_regression
from tools import randomline
from tools import target_function
from tools import target_vector

def compute_Eval(wlin, X, y):
    'fraction of in sample points which got classified incorrectly from Validation data set'
    N = len(y)
    g_vector = sign(dot(X,wlin))
    
    vEin = g_vector - y
    nEin = 0
    for i in range(len(vEin)):
        if vEin[i]!= 0: nEin = nEin + 1

    return nEin / (len(vEin) *1.0)

def compute_Eout_from_data(w,t_set_out,N_points):
    'number of out-of-sample points misclassifed/total number of out-of-sample points from data'
    
    X_matrix = input_data_matrix(t_set_out)
    y_vector = target_vector(t_set_out)
    g_vector = dot(X_matrix,w)
    for i in range(len(g_vector)):
        g_vector[i] = sign(g_vector[i])
    
    vEout = g_vector - y_vector
    nEout = 0
    for i in range(len(vEout)):
        if vEout[i]!=0:
            nEout = nEout + 1
    Eout = nEout/(len(vEout)*1.0)
    return Eout

def run_validation(indata_train,indata_val,outdata):
    dict_t_set = {}
    dict_wlin = {}
    dict_t_setval = {}
    dict_yval = {}
    dict_Xval = {}
    dict_Eval = {}
    dict_Eout = {}
    dict_outdata = {}

    #t_set train with transformation
    for i in range(3,8):
        dict_t_set[i] = transform_t_set(indata_train,i)

    #linear regression
    for i in range(3,8):
        t_set = dict_t_set[i]
        size_t_set = len(t_set)
        wlin,X,y = linear_regression(size_t_set,t_set)
        dict_wlin[i] = wlin

    #t_set validation
    for i in range(3,8):
        t_setval = transform_t_set(indata_val,i)
        dict_t_setval[i] = t_setval

    for i in range(3,8):
        t_setval = dict_t_setval[i]

        yval = target_vector(t_setval)
        dict_yval[i] = yval

        Xval = input_data_matrix(t_setval)
        dict_Xval[i] = Xval

    #Eval
    for i in range(3,8):
        wlin = dict_wlin[i]
        Xval = dict_Xval[i]
        yval = dict_yval[i]

        Eval = compute_Eval(wlin,Xval,yval)
        dict_Eval[i] = Eval

    #Eout
    for i in range(3,8):
        curr_outdata = transform_t_set(outdata,i)
        dict_outdata[i] = curr_outdata

    for i in range(3,8):
        wlin = dict_wlin[i]
        curr_outdata = dict_outdata[i]
        eout = compute_Eout_from_data(wlin,curr_outdata,len(curr_outdata))
        dict_Eout[i] = eout
    
    for i in range(3,8):
        Eval = dict_Eval[i]
        Eout = dict_Eout[i]

        print 'Eval for k = %s is: %s'%(i,Eval)
        print 'Eout for k = %s is: %s'%(i,Eout)
        print ''    

def transform_t_set(data,filter_k):
    '''Transform a dataset (data) following a filter (filter_k).
    Transformation vector: 1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|
    The filter k acts in the transformation vector.
    Example: transform_t_set(data,3) means that the transformation vector is [1, x1, x2]
    - data: data set with format [[x1,x2],y]
    - filter_k: index until which the transformation is performed.
    _ returns: transformed data set.'''
    trans_data=[]
    for i in range(len(data)):
        x1 = data[i][0][0]
        x2 = data[i][0][1]
        y = data[i][1]
        tX = [1,x1,x2,x1**2,x2**2,x1*x2,abs(x1-x2),abs(x1+x2)]

        tX = tX[:filter_k+1]

        trans_data.append([ tX,y ])
    return trans_data
def cross_validation(p):
    p = float(p)
    return 4/(1+p)**2 + 4/(p-1)**2 + 1 
           
def test_cv():  
    ps = [ sqrt(sqrt(3)+4), sqrt(sqrt(3)-1), sqrt(9+4*sqrt(6)), sqrt(9 - sqrt(6)) ]
    for p in ps:
        print cross_validation(p)

def computeEout_svm(f,w):
    return evaluate_diff_f_g(f,w)

def run_pla_vs_svm(nbruns = 1, N = 10):
    solvers.options['show_progress'] = False
    
    d = []
    l = 0
    f = 0
    t_set = []
    y = []
    svm_vs_pla = []
    for i in range(nbruns):
        onBothSides = False
        while(not onBothSides):
            d = data(N)
            l = randomline()
            f = target_function(l)
            t_set = build_training_set(d,f)
            y = target_vector(t_set)
            if (1 in y) and (-1 in y):
                onBothSides = True
            else:
                onBothSides = False
        w = [0,0,0]
        w_pla,iteration = PLA(N,w,f,t_set)
        plaEout = evaluate_diff_f_g(f,w_pla)
        X_matrix = input_data_matrix(t_set)
        dimension = len(X_matrix[0])
        #identity matrix of size dim X dim matrix x,I,J,typecode double
        P = spmatrix(1, range(dimension), range(dimension), tc='d')
        #vector of zeros of size dim, typecode double
        q = matrix([0]*(dimension), tc='d')

        mat = []
        for t in t_set:
            y = t[1]
            temp = [x * -1.0*y for x in t[0]]
            mat.append(temp) 
        
        G = matrix(mat, tc='d')
        G = G.trans()
        # vectors of -1 of size t_set
        h = matrix([-1]*len(t_set), tc='d')
        #http://abel.ee.ucla.edu/cvxopt/examples/tutorial/qp.html
        qp_sln = solvers.qp(P, q, G, h)
        wsvm = list(qp_sln['x'])
        # number of support vectors you can get at each run
        count_sv = 0
        for t in t_set:
            wsvm = array(wsvm)
            x = array(t[0])
            y = t[1]
            res = fabs(y*dot(wsvm,x)-1)
            if res < 0.001:
                count_sv = count_sv + 1
        #print count_sv
        # Eout of svm
        svmEout = computeEout_svm(f,wsvm)
        #print 'svmEout: %s'%svmEout
        if(svmEout < plaEout):
            svm_vs_pla.append([True,count_sv])
        else:
            svm_vs_pla.append([False,count_sv])

    print "svm win pla %f" % (len(filter(lambda a: a[0] is True, svm_vs_pla))*1.0/N) 
    percent_svm_won = len([r[0] for r in svm_vs_pla if r[0] is True])*1.0/len(svm_vs_pla)
    print "question 9: svm beat pla %f percent of the time" % (percent_svm_won*100)

    avg_sv = sum([a[1] for a in svm_vs_pla])*1.0/len(svm_vs_pla) 
    print "avg sv:", avg_sv 
def tests():
    print '-1-'
    print '-2-'
    indata  = data_from_file('in.dta')
    #split in.dta into training(25) and validation(10)
    indata_train = indata[:25]
    indata_val = indata[25:]
    outdata = data_from_file('out.dta')
    # train on 25 examples
    # validate on 10 examples
    run_validation(indata_train,indata_val,outdata)
    print '-3-'
    print '-4-'
    print '-5-'
    run_validation(indata_val,indata_train,outdata)
    print '-6-'
    print '-7-'
    test_cv()
    print '-8-'
    run_pla_vs_svm(1000,10)
    print '-9-'
    print '-10-'
    #run_pla_vs_svm(1000,100)

