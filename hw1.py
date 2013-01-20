
'''The Perceptron Learning Algorithm
In this problem, you will create your own target function f and data set D to see
how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
problem, and assume X = [-1; 1] x [-1; 1] with uniform probability of picking each
x appartien X .
In each run, choose a random line in the plane as your target function f (do this by
taking two random, uniformly distributed points in [-1; 1] x [-1; 1] and taking the
line passing through them), where one side of the line maps to +1 and the other maps
to -1. Choose the inputs xn of the data set as random points (uniformly in X ), and
evaluate the target function on each xn to get the corresponding output yn.'''

'''Part 1
Take N = 10. Run the Perceptron Learning Algorithm to nd g and measure
the dierence between f and g as Pr(f(x) =6 g(x)) (you can either calculate
this exactly, or approximate it by generating a suciently large separate set of
points to evaluate it). Repeat the experiment for 1000 runs (as specied above)
and take the average. Start the PLA with the weight vector w being all zeros,
and at each iteration have the algorithm choose a point randomly from the set
of misclassifed points.
How many iterations does it take on average for the PLA to converge for N = 10
training points? Pick the value closest to your results (again, closest is the
answer that makes the expression jyour answer  given optionj closest to 0).'''
from random import uniform
from random import randint

from tools import data
from tools import randomline
from tools import target_function
from tools import build_training_set
from tools import sign


def build_misclassified_set(t_set,w):
    '''returns a tuple of index of t_set items
    such that t_set[index] is misclassified <=> yn != sign(w*point)'''
    res = tuple()
            
    for i in range(len(t_set)):
        point = t_set[i][0]
        s = h(w,point)
        yn = t_set[i][1]
        if s != yn:
            res = res + (i,)
    return res

def h(w,x):
    'Hypothesis function returns w0 x0 + w1 x1 ... + wn xn'
    res = 0
    for i in range(len(x)):
        res = res + w[i]*x[i]
    return sign(res)

def PLA(N_points = 10):
    ''' 
    Returns: (t_set,w,iteration,f)
    -t_set: item of t_set is: [[vector_x], y]
    -w: vector of same dimention as vector_x of weights
    -iteration: Number of iterations needed for convergence
    -f: target lambda function f
    '''
    N = N_points
    iteration = 0
    # create random contelation of points () in interval [-1,1]
    # create random target function
    # build training set
    
    d = data(N)
    l = randomline()
    f = target_function(l)
    t_set = build_training_set(d,f)

    w = [0,0,0] # weight vector w0 , w1, w2  

    #iterate Perceptron Algorithm
    iterate = True
    count = 0
    while iterate:
        iteration = iteration + 1
        #pick a misclasified point from misclassified set
        misclassified_set = build_misclassified_set(t_set,w)
        # if there are no misclassified points break iteration weight are ok.
        if len(misclassified_set)==0:break
        index = randint(0,len(misclassified_set)-1)
        p = misclassified_set[index]
        point = t_set[p][0]

        s = h(w,point)
        yn = t_set[p][1]

        # update weights if misclassified
        if s != yn:
            xn = point
            w[0] = w[0] + yn*xn[0]
            w[1] = w[1] + yn*xn[1]
            w[2] = w[2] + yn*xn[2]
    return t_set,w,iteration,f

def evaluate_diff_f_g(f,w):
    'Returns the average of difference between f and g (g is equivalent as vector w )'
    count = 0
    limit = 100
    diff = 0
    while count < limit:
        count = count + 1
        # generate random point as out of sample data
        x = uniform(-1,1)
        y = uniform(-1,1)
        vector = [1,x,y]

        sign_f = sign(f(x),y)
        sign_g = h(w,vector)
        # check result and count if difference between target function f and hypothesis function g
        if sign_f != sign_g: diff = diff + 1

    return diff/(count*1.0)
        
    
def run_PLA(N_samples,N_points):
    samples = []# vector of 1 clasified, 0 misclassified
    iterations = []#vector of iterations needed for each PLA
    b_misclassified = False
    diff = []#vector of difference average between f and g

    for i in range(N_samples):
        # run PLA in sample
        t_set,w,iteration,f = PLA(N_points)
        iterations.append(iteration)
        # check if points are classified or not
        for i in range(len(t_set)):
            point = t_set[i][0]
            s = h(w,point)
            yn = t_set[i][1]
            if yn != s:
                samples.append(0)
                b_misclassified = True
                break
        # check difference between f and g
        diff.append(evaluate_diff_f_g(f,w))
        if not b_misclassified: samples.append(1)

        b_misclassified = False

    print 'number of samples misclassified: %s ' % samples.count(0)
    print 'number of classified samples: %s ' % samples.count(1)
    print 'number of iteration avg: %s ' % (str(sum(iterations)/len(iterations)*1.0))
    print 'average of difference in function g: %s' % ( sum(diff)/(len(diff)*1.0) )

    
