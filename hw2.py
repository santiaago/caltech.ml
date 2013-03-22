from numpy import array
from numpy import dot
from numpy import sign
from numpy import transpose
from numpy.linalg import pinv # pseudo inverse aka dagger
from numpy.linalg import norm

from random import randint

from hw1 import PLA

from tools import build_training_set
from tools import build_training_set_fmultipleparams
from tools import data
from tools import input_data_matrix
from tools import linear_regression
from tools import pseudo_inverse
from tools import print_avg
from tools import randomline
from tools import sign
from tools import target_function
from tools import target_vector

N_COINS = 1000
N_TOSS = 10
N_EXPERIMENT = 100000
HEAD = 0
TAILS = 1

#--------------------------------------------------------------------------
#Hoeffding

def hoeffding_inequality():
    '''average experiment on N_EXPERIMENT times
    toss the N_COINS N_TOSS times
    get c1, crand and cmin
    print average of each
    '''
    table_v1 = []
    table_vrand = []
    table_vmin = []
    for experiment in range(N_EXPERIMENT):

        coins = [flip_coin() for c in range(N_COINS)]
        
        # C1 the first coin you toss
        c1 = coins[0]
        # Crand is a coin chosen randomly from the N_COINS
        index_rand = randint(0,len(coins)-1)
        crand = coins[index_rand]
        # Cmin is a coin which had the minimum frequency of heads
        cmin = compute_min_frec_heads(coins)
        #print_coins(c1,crand,cmin)

        table_v1.append(fractionOfHeads(c1))
        table_vrand.append(fractionOfHeads(crand))
        table_vmin.append(fractionOfHeads(cmin))
    
    # compute averages.
    v1_avg = sum(table_v1)/len(table_v1)
    vrand_avg = sum(table_vrand)/len(table_vrand)
    vmin_avg = sum(table_vmin)/len(table_vmin)
    print_vs(v1_avg,vrand_avg,vmin_avg)

    '''Results:
    v1 = 0.500339
    vrand = 0.500511
    vmin = 0.03751'''

def fractionOfHeads(c):
    'fractions of Heads in list c'
    return c.count(HEAD)/(N_TOSS*1.0)

def compute_min_frec_heads(coins):
    'minimum frecuency of heads in list coins '
    
    f_heads = [fractionOfHeads(c) for c in coins]
    
    id_min_frec_heads = f_heads.index(min(f_heads))
    
    return coins[id_min_frec_heads]

def print_coins(c1,crand,cmin):
    'print c1, crand and cmin as well as their fraction of Heads'
    print 'c1 = %s' % (str(c1))
    print 'v1 = %s' % (str(fractionOfHeads(c1)))

    print 'crand = %s' %(str(crand))
    print 'vrand = %s' %(str(fractionOfHeads(crand)))

    print 'cmin = %s' %(str(cmin))
    print 'vmin = %s' %(str(fractionOfHeads(cmin)))

def print_vs(v1,vrand,vmin):
    'print v1 vrand and vmin'
    print 'v1 = %s' % (str(v1))
    print 'vrand = %s' % (str(vrand))
    print 'vmin = %s' % (str(vmin))

def flip_coin(n = N_TOSS):
    'list of n experiment between 0 and 1 randomly'
    return  [randint(0,1) for i in range(n)]

#--------------------------------------------------------------------------
#Linear regresion

verbose_lr = False

def run_linear_regression(N_samples,N_points):
    '''runs on N_samples and with N_points a linear regression
    computes Ein by average of the samples as well as Eout
    '''
    print 'running Linear Regression on %s samples' %str(N_samples)
    print 'Each sample has %s data points' %str(N_points)

    Ein_avg = []
    Eout_avg = []

    for i in range(N_samples):

        d = data(N_points)
        l = randomline()
        f = target_function(l)
        t_set = build_training_set(d,f)

        wlin,X,y = linear_regression(N_points,t_set)

        Ein = compute_Ein(wlin,X,y)
        Ein_avg.append(Ein)

        Eout = compute_Eout(wlin,f,N_points)
        Eout_avg.append(Eout)
        
    print_avg('Ein',Ein_avg)
    print_avg('Eout',Eout_avg)

def run_lr_and_pla(N_samples, N_points):
    '''runs on N_samples and with N_points a linear regresion
    then from the weight vector runs PLA algorithm
    compute the average number of iterations of PLA with this w vector
    '''
    print 'running Linear Regression on %s samples' %N_samples
    print 'Each samples has %s data points' %N_points
    
    iteration_avg = []
    for i in range(N_samples):

        d = data(N_points)
        l = randomline()
        f = target_function(l)
        t_set = build_training_set(d,f)
        
        wlin,X,y = linear_regression(N_points,t_set)
        
        w_pla,iteration = PLA(N_points,wlin,f,t_set)
        iteration_avg.append(iteration)
    
    print_avg('Number of iterations',iteration_avg)
        
def compute_Eout(wlin,f,N_points):
    'number of out-of-sample points misclassifed / total number of out-of-sample points'
    
    d = data(N_points)
    t_set = build_training_set(d,f)
    
    X_matrix = input_data_matrix(t_set)
    y_vector = target_vector(t_set)
    
    g_vector = dot(X_matrix,wlin)
    for i in range(len(g_vector)):
        g_vector[i] = sign(g_vector[i])
    
    vEout = g_vector - y_vector
    nEout = 0
    for i in range(len(vEout)):
        if vEout[i]!=0:
            nEout = nEout + 1
    Eout = nEout/(len(vEout)*1.0)
    return Eout

def compute_Ein(wlin, X, y):
    'fraction of in sample points which got classified incorrectly'
    N = len(y)
    g_vector = sign(dot(X,wlin))
    
    vEin = g_vector - y
    nEin = 0
    for i in range(len(vEin)):
        if vEin[i]!= 0: nEin = nEin + 1

    return nEin / (len(vEin) *1.0)
    
#--------------------------------------------------------------------------
#Nonlinear Transformation

def generate_t_set(N,f=None):
    '''
    Generate a training set of N = 1000 points on X = [1; 1] * [1; 1] with uniform
    probability of picking each x that belongs X . Generate simulated noise by 
    fipping the sign of a random 10% subset of the generated training set
    '''
    d = data(N)
    if f is None:
        f = lambda x: sign(x[0]**2 + x[1]**2 -0.6)
    
    t_set = build_training_set_fmultipleparams(d,f)
    t_set = t_set_errorNoise(t_set,N/10)
        
    return t_set,f

def t_set_errorNoise(t_set, Nnoise):
    'introduce N% noise in the sign of the training set'
    for i in range(Nnoise):
        j = randint(0,Nnoise-1)
        t_set[j][1] = t_set[j][1]*-1
    return t_set

def run_nonlinear_transformation(N_samples, N_points):
    '''use N_samples to have a consistent result
    create a trainng set (1; x1; x2) from a constalation on N_points
    runs linear regration from training set
    computes Ein and averages it through all the samples
    transform the training set following (1; x1; x2; x1x2; x1^2; x2^2)
    run linear transformation on this transformed training set
    compute Ein of transformed t_set and average through all the samples
    create a hypothesis vector from the weight vector and the X matrix of the t_set transformed
    Average for each function g the difference between the hypothesis vector and the function
    finaly compute Eout from the f (target function) and the weight vector from training set that was not transformed
    '''
    Ein_avg = []
    Eout_avg = []
    Eintrans_avg = []
    EdiffA = []
    EdiffB = []
    EdiffC = []
    EdiffD = []
    EdiffE = []

    for i in range(N_samples):

        t_set,f = generate_t_set(N_points)
        wlin,X,y = linear_regression(N_points,t_set)
        Ein = compute_Ein(wlin, X, y)
        Ein_avg.append(Ein)

        #transform the training data into the following nonlinear feature vector:
        #(1; x1; x2; x1x2; x1^2; x2^2)
        t_set_trans = transform_t_set(t_set)
        wtrans,Xtrans,ytrans = linear_regression(N_points,t_set_trans)
        Eintrans = compute_Ein(wtrans,Xtrans,ytrans)
        Eintrans_avg.append(Eintrans)
    
        h_vector =sign(dot(Xtrans,wtrans))
        gA_vector = compute_g_vector(t_set_trans,'a')
        Ediff_a = compute_avg_difference(h_vector,gA_vector)
        EdiffA.append(1-Ediff_a)
        
        gB_vector = compute_g_vector(t_set_trans,'b')
        Ediff_b = compute_avg_difference(h_vector,gB_vector)
        EdiffB.append(1-Ediff_b)

        gC_vector = compute_g_vector(t_set_trans,'c')
        Ediff_c = compute_avg_difference(h_vector,gC_vector)
        EdiffC.append(1-Ediff_c)
        
        gD_vector = compute_g_vector(t_set_trans,'d')
        Ediff_d = compute_avg_difference(h_vector,gD_vector)
        EdiffD.append(1-Ediff_d)
        
        gE_vector = compute_g_vector(t_set_trans,'e')
        Ediff_e = compute_avg_difference(h_vector,gE_vector)
        EdiffE.append(1-Ediff_e)

        Eout = compute_Eout_nonlineartrans(wtrans,f,N_points)
        Eout_avg.append(Eout)

    print_avg('Ein',Ein_avg)
    print_avg('Ein Transformed',Eintrans_avg)
    print_avg('P of agreeing A',EdiffA)
    print_avg('P of agreeing B',EdiffB)
    print_avg('P of agreeing C',EdiffC)
    print_avg('P of agreeing D',EdiffD)
    print_avg('P of agreeing E',EdiffE)
    print_avg('Eout',Eout_avg)

def compute_Eout_nonlineartrans(w,f,N_points):
    'number of out-of-sample points misclassifed / total number of out-of-sample points'
    
    # generate N fresh points (f will not change) with noise
    t_set,f = generate_t_set(N_points,f)
    t_set_trans = transform_t_set(t_set)
    
    X_matrix = input_data_matrix(t_set_trans)
    y_vector = target_vector(t_set_trans)

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

def transform_t_set(t_set):
    '''returns a training set built from a simple training [1,x1,x2] set and with
    the following transformation: [1, x1, x2, x1*x2, x1^2, x2^2]
    '''
    t_set_trans = []
    for i in range(len(t_set)):
        x1 = t_set[i][0][1]
        x2 = t_set[i][0][2]
        tX = [1,x1,x2, x1*x2, x1**2, x2**2]
        t_set_trans.append([ tX , t_set[i][1] ])

    return t_set_trans

def compute_avg_difference(v1,v2):
    'from to vectors compute the average number of differences between them'
    vDiff = v1 - v2
    nE = 0
    for i in range(len(vDiff)):
        if vDiff[i]!= 0: nE = nE + 1
    return nE / (len(vDiff) *1.0)

def compute_g_vector(t_set,g_f):
    'collect the values of the chosen function g and the provided training set'
    g = []
    for t in t_set:
        x1 = t[0][1]
        x2 = t[0][2]
        if g_f == 'a': g.append(gA(x1,x2))
        if g_f == 'b': g.append(gB(x1,x2))
        if g_f == 'c': g.append(gC(x1,x2))
        if g_f == 'd': g.append(gD(x1,x2))
        if g_f == 'e': g.append(gE(x1,x2))

    return g

# G functions to compare to f.
def gA(x1,x2):
    return sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1**2 + 1.5*x2**2)
def gB(x1,x2):
    return sign(-1  -0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1**2 + 15*x2**2)
def gC(x1,x2):
    return sign(-1 -0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 15*x1**2 + 1.5*x2**2)
def gD(x1,x2):
    return sign(-1 -1.5*x1 + 0.08*x2 + 0.13*x1*x2 + 0.05*x1**2 + 0.05*x2**2)
def gE(x1,x2):
    return sign(-1 -0.05*x1 + 0.08*x2 + 1.5*x1*x2 + 0.15*x1**2 + 0.15*x2**2)

def tests():
    #-1-2
    #hoeffding_inequality()
    #3
    #4
    #5-6
    run_linear_regression(1000,100)
    #-7
    run_lr_and_pla(1000, 10)
    #8-9-10
    run_nonlinear_transformation(1000, 1000)
