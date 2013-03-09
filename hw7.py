
from tools import linear_regression
from tools import target_vector
from tools import input_data_matrix
from tools import sign

from hw1 import evaluate_diff_f_g

from math import sqrt

from numpy import dot
from numpy import sign

def getData(filename):
    datafile = open(filename, 'r')
    data = []
    for line in datafile:
        split = line.split()
        x1 = float(split[0])
        x2 = float(split[1])
        y = float(split[2])
        data.append([ [x1,x2],y ])
    return data

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
    'number of out-of-sample points misclassifed / total number of out-of-sample points from data'
    
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
    #t_set train
    t_set3 = transform_t_set(indata_train,3)
    t_set4 = transform_t_set(indata_train,4)
    t_set5 = transform_t_set(indata_train,5)
    t_set6 = transform_t_set(indata_train,6)
    t_set7 = transform_t_set(indata_train,7)
    
    #lnr
    wlin3,X,y = linear_regression(len(t_set3),t_set3)
    wlin4,X,y = linear_regression(len(t_set4),t_set4)
    wlin5,X,y = linear_regression(len(t_set5),t_set5)
    wlin6,X,y = linear_regression(len(t_set6),t_set6)
    wlin7,X,y = linear_regression(len(t_set7),t_set7)

    #t_set validation
    t_setVal3 = transform_t_set(indata_val,3)
    t_setVal4 = transform_t_set(indata_val,4)
    t_setVal5 = transform_t_set(indata_val,5)
    t_setVal6 = transform_t_set(indata_val,6)
    t_setVal7 = transform_t_set(indata_val,7)

    yVal3 = target_vector(t_setVal3)
    XVal3 = input_data_matrix(t_setVal3)
    
    yVal4 = target_vector(t_setVal4)
    XVal4 = input_data_matrix(t_setVal4)
    
    yVal5 = target_vector(t_setVal5)
    XVal5 = input_data_matrix(t_setVal5)
    
    yVal6 = target_vector(t_setVal6)
    XVal6 = input_data_matrix(t_setVal6)
    
    yVal7 = target_vector(t_setVal7)
    XVal7 = input_data_matrix(t_setVal7)
    #Eval
    Eval3 = compute_Eval(wlin3,XVal3,yVal3)
    Eval4 = compute_Eval(wlin4,XVal4,yVal4)
    Eval5 = compute_Eval(wlin5,XVal5,yVal5)
    Eval6 = compute_Eval(wlin6,XVal6,yVal6)
    Eval7 = compute_Eval(wlin7,XVal7,yVal7)

    #Eout
    outdata3 = transform_t_set(outdata,3)
    outdata4 = transform_t_set(outdata,4)
    outdata5 = transform_t_set(outdata,5)
    outdata6 = transform_t_set(outdata,6)
    outdata7 = transform_t_set(outdata,7)

    Eout3 = compute_Eout_from_data(wlin3,outdata3,len(outdata3))
    Eout4 = compute_Eout_from_data(wlin4,outdata4,len(outdata4))
    Eout5 = compute_Eout_from_data(wlin5,outdata5,len(outdata5))
    Eout6 = compute_Eout_from_data(wlin6,outdata6,len(outdata6))
    Eout7 = compute_Eout_from_data(wlin7,outdata7,len(outdata7))

    print 'Eval for k = 3 is: %s'%Eval3
    print 'Eout for k = 3 is: %s'%Eout3
    print ''
    print 'Eval for k = 4 is: %s'%Eval4
    print 'Eout for k = 4 is: %s'%Eout4
    print ''
    print 'Eval for k = 5 is: %s'%Eval5
    print 'Eout for k = 5 is: %s'%Eout5
    print ''
    print 'Eval for k = 6 is: %s'%Eval6
    print 'Eout for k = 6 is: %s'%Eout6
    print ''
    print 'Eval for k = 7 is: %s'%Eval7
    print 'Eout for k = 7 is: %s'%Eout7
    

def transform_t_set(data,filter_k):
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
    from tools import data
    from tools import randomline
    from tools import target_function
    from tools import build_training_set
    from tools import target_vector
    
    from hw1 import PLA
    #cvxopt Python Software for Convex Optimization
    #from http://abel.ee.ucla.edu/cvxopt/
    import cvxopt
    from cvxopt import solvers
    solvers.options['show_progress'] = False
    from numpy import dot
    from numpy import array

    from math import fabs
    
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
        P = cvxopt.spmatrix(1, range(dimension), range(dimension), tc='d')
        #vector of zeros of size dim, typecode double
        q = cvxopt.matrix([0]*(dimension), tc='d')

        mat = []
        for t in t_set:
            y = t[1]
            temp = [x * -1.0*y for x in t[0]]
            mat.append(temp) 
        
        G = cvxopt.matrix(mat, tc='d')
        G = G.trans()
        # vectors of -1 of size t_set
        h = cvxopt.matrix([-1]*len(t_set), tc='d')
        #http://abel.ee.ucla.edu/cvxopt/examples/tutorial/qp.html
        qp_sln = cvxopt.solvers.qp(P, q, G, h)
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

    print 'after run: '
    print "svm win pla %f" % (len(filter(lambda a: a[0] is True, svm_vs_pla))*1.0/N)
    print "svm win pla %f" % (len(filter(lambda a: a[0] is True, svm_vs_pla))*1.0/len(svm_vs_pla))
    print "svm %f"%(len(filter(lambda a:a[0] is True,svm_vs_pla)))
    print "len svm_vs_pla %s"%len(svm_vs_pla)
    percent_svm_won = len([r[0] for r in svm_vs_pla if r[0] is True])*1.0/len(svm_vs_pla)
    print "question 9: svm beat pla %f percent of the time" % percent_svm_won 

    avg_sv = sum([a[1] for a in svm_vs_pla])*1.0/len(svm_vs_pla) 
    print "avg sv:", avg_sv 
def tests():
    print '-1-'
    print '-2-'
    indata  = getData('in.dta')
    #split in.dta into training(25) and validation(10)
    indata_train = indata[:25]
    indata_val = indata[25:]
    outdata = getData('out.dta')
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
    #run_pla_vs_svm(1000,10)
    print '-9-'
    print '-10-'
    run_pla_vs_svm(1000,100)

