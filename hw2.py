
'''
Heads = 0
Tails = 1
'''
N_COINS = 1000
N_TOSS = 10
N_EXPERIMENT = 100000

from random import randint

#--------------------------------------------------
#Hoeffding inequality
def main():
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
        # append values of fractions for average after experiment
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
    return c.count(0)/(N_TOSS*1.0)

def compute_min_frec_heads(coins):
    'min frec of head (head = 0 )'
    
    f_heads = [fractionOfHeads(c) for c in coins]
    
    id_min_frec_heads = f_heads.index(min(f_heads))
    
    return coins[id_min_frec_heads]

def print_coins(c1,crand,cmin):
    print 'c1 = %s' % (str(c1))
    print 'v1 = %s' % (str(fractionOfHeads(c1)))

    print 'crand = %s' %(str(crand))
    print 'vrand = %s' %(str(fractionOfHeads(crand)))

    print 'cmin = %s' %(str(cmin))
    print 'vmin = %s' %(str(fractionOfHeads(cmin)))

def print_vs(v1,vrand,vmin):
    print 'v1 = %s' % (str(v1))
    print 'vrand = %s' % (str(vrand))
    print 'vmin = %s' % (str(vmin))

def flip_coin(n = N_TOSS):
    return  [randint(0,1) for i in range(n)]

#--------------------------------------------------------------------------
#Linear regresion

from tools import data
from tools import randomline
from tools import target_function
from tools import build_training_set
from tools import sign

from numpy import array
from numpy import transpose
from numpy.linalg import pinv as pinv # pseudo inverse aka dagger
from numpy.linalg import norm
from numpy import dot

verbose_lr = False
def linear_regression(N_points = 100):
    d = data(N_points)
    l = randomline()
    f = target_function(l)

    t_set = build_training_set(d,f)

    y_vector = target_vector(t_set)
    X_matrix = input_data_matrix(t_set)
    X_pseudo_inverse = pseudo_inverse(X_matrix)

    if verbose_lr:
        print 'y: %s'% y_vector
        print '----------'
        print 'X: %s' %X_matrix
        print '----------'
        print 'X pseudo inverse: %s '%X_pseudo_inverse
        print '----------'
        print 'wlin: %s' %(dot(X_pseudo_inverse,y_vector) )
        print '----------'

    return dot(X_pseudo_inverse,y_vector),X_matrix,y_vector

def run_linear_regression(N_samples,N_points):
    print 'running Linear Regression on %s samples' %str(N_samples)
    print 'Each sample has %s data points' %str(N_points)

    Ein_avg = []
    for i in range(N_samples):
        wlin,X,y = linear_regression(N_points)
        Ein = compute_Ein(wlin,X,y)
        Ein_avg.append(Ein)
        if verbose_lr:
            print 'Ein: %s '% Ein
    print 'Average Ein: %s' %(sum(Ein_avg)/(N_samples*1.0))

def compute_Ein(wlin, X, y):
    'fraction of in sample points which got classified incorrectly'
    N = len(y)
    g_vector = dot(X,wlin) #X * wlin
    for i in range(len(g_vector)):
        g_vector[i] = sign(g_vector[i])

    vEin = g_vector - y
    nEin = 0
    for i in range(len(vEin)):
        if vEin[i]!= 0:
            nEin = nEin + 1

    Ein = nEin / (len(vEin) *1.0)
    return Ein
    
def target_vector(t_set):
    y = array([t[1] for t in t_set])
    return y

def input_data_matrix(t_set):
    X = array([t[0] for t in t_set])
    return X

def pseudo_inverse(X):
    return pinv(X)
