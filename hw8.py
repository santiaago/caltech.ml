from numpy import array
from numpy import dot
from numpy import eye
from numpy import size
from numpy import zeros
from numpy.linalg import pinv

from svmutil import svm_predict
from svmutil import svm_train

from random import shuffle

from tools import linear_regression

from hw2 import compute_Ein
from hw2 import transform_t_set

def getDataFeatures(filename):
    'from file return list of items in the form of [digit, symmerty, intensity]'
    datafile = open(filename, 'r')
    data = []
    for line in datafile:
        split = line.split()
        digit = float(split[0])
        symmetry = float(split[1])
        intensity = float(split[2])
        data.append([ float(digit),float(symmetry),float(intensity) ])
    return data

def getDataOneVsOne(dataset,d1,d2):
    'binary classifier: d1 will be class 1, d2 will be class -1, other digits are discarded'
    new_d = []
    for d in dataset:
        digit, symmerty, intensity = d
        if digit in (d1,d2):
            if digit == d1:
                digit = 1.0
            else:
                digit = -1.0
            new_d.append([float(digit),float(symmerty),float(intensity)])
    return new_d

def getDataOneVsAll(dataset,d1):
    'binary classifier: d1 will be class 1, other digits will be class -1'
    new_d = []
    for d in dataset:
        digit,symmerty,intensity = d
        if digit == d1:
            digit = 1.0
        else:
            digit = -1.0
        new_d.append([float(digit),float(symmerty),float(intensity)])
    return new_d

def get_svm_vector_format(data):
    '''This method formats data in lib svm format: 
    Y is a 1 dimention vector, 
    X is a dictionary of key 1: symmerty, key 2: intensity'''

    Y=[]
    X=[]
    for i in range(len(data)):
	Y.append(data[i][0])
	X.append({1 : data[i][1], 2:data[i][2]})
    return X,Y

def run_poly_kernel(dTrain,dTest):
    'Work on polynomial kernel and compare different classifiers'

    print '--run poly kernel--'
    dTrain_all = []
    dTest_all = []
    X_train_all = []
    Y_train_all = []
    # get data for all One vs All and format X and Y
    for i in range(10):
        dTrain_current = getDataOneVsAll(dTrain,i)
        dTrain_all.append(getDataOneVsAll(dTrain,i))
        dTest_all.append(getDataOneVsAll(dTest,i))
        X_train,Y_train = get_svm_vector_format(dTrain_current)
        X_train_all.append(X_train)
        Y_train_all.append(Y_train)

    # train all models
    print '--starting training--'
    model_all =[]
    # Type = Polynomial. Degree = 2. Gamma 1. Coef = 1. C = 0.01. be quiet.
    options = '-t 1 -d 2 -g 1 -r 1 -c 0.01 -q'
    for i in range(10):
        model = svm_train(Y_train_all[i],X_train_all[i],options)
        model_all.append(model)
    # test even models
    print '--testing model --'
    max_Ein_q2 = 0
    max_Ein_q2_index = -1
    for i in [0,2,4,6,8]:
        plabel, paccuracy,pvals = svm_predict(Y_train_all[i],X_train_all[i],model_all[i])
        ein = 100 - paccuracy[0]
        if ein > max_Ein_q2:
            max_Ein_q2 = ein
            max_Ein_q2_index = i
    # test odd models
    min_Ein_q3 = 100
    min_Ein_q3_index = -1
    for i in [1,3,5,7,9]:
        plabel, paccuracy,pvals = svm_predict(Y_train_all[i],X_train_all[i],model_all[i])
        ein = 100 - paccuracy[0]
        if ein < min_Ein_q3:
            min_Ein_q3 = ein
            min_Ein_q3_index = i
    # display
    print
    print 'The classifier ''%s versus all'' has the highest Ein: %s = from the following group %s'%(max_Ein_q2_index,max_Ein_q2,[0,2,4,6,8])
    print
    print 'The classifier ''%s versus all'' has the lowest Ein: %s from the following group: %s'%(min_Ein_q3_index,min_Ein_q3,[1,3,5,7,9])
    # Comparing Models
    print
    print 'Comparing previous classifiers'
    print
    m0 = model_all[max_Ein_q2_index]
    m1 =  model_all[min_Ein_q3_index]
    print '--m0--'
    print 'Number of Support Vectors from model: %s versus all is: %s'%(str(max_Ein_q2_index),str(m0.get_nr_sv()))
    print
    print '--m1--'
    print 'Number of Support Vectors from model: %s versus all is: %s'%(str(min_Ein_q3_index),str(m1.get_nr_sv()))
    print
    print 'Difference: %s'%(abs(m0.get_nr_sv()-m1.get_nr_sv()))
   
def run_1vs5_q2_q5(dTrain,dTest):
    'Work on 1 vs 5 classifiers with Polynomial kernel by comparing degree Q'

    print '-- 1 Vs 5 with Q = 2 and Q = 5 --'
    Cs = [0.0001,0.001,0.01,0.1,1]
    # vars for polynomial of degree 5
    model_q5 = []
    Eins_q5 = []
    Eouts_q5 = []
    nb_svs_q5 = []
    # vars for polynomial of degree 2
    model_q2 = []
    Eins_q2 = []
    Eouts_q2 = []
    nb_svs_q2 = []

    # Get data and format it
    dTrain_1vs5 = getDataOneVsOne(dTrain,1,5)
    dTest_1vs5 = getDataOneVsOne(dTest,1,5)

    X_train_1vs5,Y_train_1vs5 = get_svm_vector_format(dTrain_1vs5)
    X_test_1vs5,Y_test_1vs5 = get_svm_vector_format(dTest_1vs5)
    
    # run Train and Test on all Cs
    for c in Cs:
        # Type = Polynomial. Degree = 5. Gamma 1. Coef = 1. C = Cs[i]. be quiet.
        options = '-t 1 -d 5 -g 1 -r 1 -c '+str(c) + ' -q'
        # Train
        m = svm_train(Y_train_1vs5,X_train_1vs5,options)
        # Test in sample
        plabel, paccuracy,pvals = svm_predict(Y_train_1vs5,X_train_1vs5,m)
        # Test out of sample
        plabel, paccuracy_test,pvals = svm_predict(Y_test_1vs5,X_test_1vs5,m)

        nb_svs_q5.append(m.get_nr_sv())
        Eins_q5.append(100 - paccuracy[0])
        Eouts_q5.append(100 - paccuracy_test[0])
        model_q5.append(m)

        # Type = Polynomial. Degree = 2. Gamma 1. Coef = 1. C = Cs[i]. be quiet.
        options = '-t 1 -d 2 -g 1 -r 1 -c '+str(c) +' -q'
        # Train
        m = svm_train(Y_train_1vs5,X_train_1vs5,options)
        # Test in sample
        plabel, paccuracy,pvals = svm_predict(Y_train_1vs5,X_train_1vs5,m)
        # Test out of sample
        plabel, paccuracy_test,pvals = svm_predict(Y_test_1vs5,X_test_1vs5,m)

        nb_svs_q2.append(m.get_nr_sv())
        Eins_q2.append(100 - paccuracy[0])
        Eouts_q2.append(100 - paccuracy_test[0])
        model_q2.append(m)
    # Display
    print
    print '-- analysis --'    
    print
    print '-- analysis Q = 2 --'
    for i in range(len(Cs)):
        print 'C = %s \tnumber of SV = %s'%(Cs[i],nb_svs_q2[i])
    print
    for i in range(len(Cs)):
        print 'C = %s \tEin = %s'%(Cs[i],Eins_q2[i])
    print
    for i in range(len(Cs)):
        print 'C = %s \tEout = %s'%(Cs[i],Eouts_q2[i])
    print
    print '-- analysis Q = 2 vs Q = 5 --'
    print 'Ein for C = %s and Q = 2 is %s'%(Cs[0],Eins_q2[0])
    print 'Ein for C = %s and Q = 5 is %s'%(Cs[0],Eins_q5[0])
    print
    print 'Number of support vectors for C = %s and Q = 2 is %s'%(Cs[1],nb_svs_q2[1])
    print 'Number of support vectors for C = %s and Q = 5 is %s'%(Cs[1],nb_svs_q5[1])
    print
    print 'Ein for C = %s and Q = 2 is %s'%(Cs[2],Eins_q2[2])
    print 'Ein for C = %s and Q = 5 is %s'%(Cs[2],Eins_q5[2])
    print
    print 'Eout for C = %s and Q = 2 is %s'%(Cs[4],Eouts_q2[4])
    print 'Eout for C = %s and Q = 5 is %s'%(Cs[4],Eouts_q5[4])

def run_cross_validation(dTrain,dTest):
    'Work with Polynomal kernel with cross validation'
    print '--run_cross_validation--'

    print '-- 1 versus 5 with Q = 2 and Cross Validation--'

    Cs = [0.0001,0.001,0.01,0.1,1]
    Ecvs = [[],[],[],[],[]]
    
    print '-- Train and Test --'

    dTrain_shuffle = dTrain
    # Try 100 runs with different partitions
    for j in range(100):
        # roll those dices
        shuffle(dTrain_shuffle)
        # Get data and formated vectors
        dTrain_1vs5 = getDataOneVsOne(dTrain_shuffle,1,5)
        X_train_1vs5,Y_train_1vs5 = get_svm_vector_format(dTrain_1vs5)
        # Try all Cs with cross validation
        for i in range(len(Cs)):
            # Type = Polynomial. Degree = 2. Gamma 1.
            # Coef = 1. C = Cs[i].Cross Validation at 10. be quiet
            options = '-t 1 -d 2 -g 1 -r 1 -c '+str(Cs[i])+ ' -v 10 -q'
            m = svm_train(Y_train_1vs5,X_train_1vs5,options)
            Ecvs[i].append(100 - m)
    # display
    print
    for i in range(len(Ecvs)):
        print 'Ecv = %s \tfor C = %s'%(sum(Ecvs[i])/100.,Cs[i])
    print
    
def run_rbf_kernel(dTrain,dTest):
    'Work on Radial Basis function'
    
    print '--run_rbf_kernel--'
    
    # Get data for 1 vs 5
    dTrain_1vs5 = getDataOneVsOne(dTrain,1,5)
    dTest_1vs5 = getDataOneVsOne(dTest,1,5)
    # Get formated vectors Train and Test
    X_train_1vs5,Y_train_1vs5 = get_svm_vector_format(dTrain_1vs5)
    X_test_1vs5,Y_test_1vs5 = get_svm_vector_format(dTest_1vs5)
    
    Eins = []
    Eouts = []
    Cs = [0.01,1.0,100.0,1.0*10**4,1.0*10**6]

    for i in range(len(Cs)):
        print 'Current C: %s'%(Cs[i])

        # Gamma = 1. Type = RBF : exp(-gamma*|u-v|^2). shrinking: off. Be quiet.
        options = '-c '+str(Cs[i])+ ' -g 1 -t 2 -h 0 -q'
        # Train
        print 'train'
        m = svm_train(Y_train_1vs5,X_train_1vs5,options)
        # Test
        print 'test In sample'
        plabel, paccuracy,pvals = svm_predict(Y_train_1vs5,X_train_1vs5,m)
        print 'test Out of sample'
        plabel, paccuracy_test,pvals = svm_predict(Y_test_1vs5,X_test_1vs5,m)
        # append Errors
        Eins.append(100 - paccuracy[0])
        Eouts.append(100 - paccuracy_test[0])
    #display
    print 
    for i in range(len(Eins)):
        print 'For C = %s \tEin = %s'%(Cs[i],Eins[i])
    print 'Min value for Ein is : %s'%(min(Eins))
    print
    for i in range(len(Eouts)):
        print 'For C = %s \tEout = %s'%(Cs[i],Eouts[i])
    print 'Min value for Eout is : %s'%(min(Eouts))

def run_reg_linear_reg_one_vs_all(dTrain,dTest):

    lda = 1.0
    for i in range(0,10):
        dTrain_current = getDataOneVsAll(dTrain,i)
        t_set = []
        # in sample
        for d in dTrain_current:
            t_set.append([[1,d[1],d[2]],d[0]])
        # out of sample
        dTest_current = getDataOneVsAll(dTest,i)
        t_setout = []
        for d in dTest_current:
            t_setout.append([[1,d[1],d[2]],d[0]])
        # in sample with no transform
        wlin,X0,y0 = linear_regression(len(t_set),t_set)
        print 'For %s vs all Ein = %s'%(i,compute_Ein(wlin,X0,y0))
        # out of sample with no transform
        wout,Xout,yout = linear_regression(len(t_setout),t_setout)
        print 'For %s vs all Eout = %s'%(i,compute_Ein(wlin,Xout,yout))
        # in sample with transform
        t_set_trans = transform_t_set(t_set)
        wtrans,Xtrans,ytrans = linear_regression(len(t_set_trans),t_set_trans)
        # out of sample with transform        
        t_setout = transform_t_set(t_setout)
        wt,xt,yt = linear_regression(len(t_setout),t_setout)
        print 'For %s vs all with transformation Eout = %s'%(i,compute_Ein(wtrans,xt,yt))

def run_reg_linear_reg_one_vs_one(dTrain,dTest):

    lda1 = 0.01
    lda2 = 1
    # 1 vs 5
    dTrain_current = getDataOneVsOne(dTrain,1,5)
    t_set = []
        # in sample
    for d in dTrain_current:
        t_set.append([[1,d[1],d[2]],d[0]])
    # out of sample
    dTest_current = getDataOneVsOne(dTest,1,5)
    t_setout = []
    t_setout2 = []
    for d in dTest_current:
        t_setout.append([[1,d[1],d[2]],d[0]])
        t_setout2.append([[1,d[1],d[2]],d[0]])
    print '--------------------------------------------------'
    print 'lambda is: %s'%(lda1)
    # in sample with no transform
    wlin,X0,y0 = linear_regression(len(t_set),t_set,lda1)
    print 'For 1 vs 5 Ein = %s'%(compute_Ein(wlin,X0,y0))
    # out of sample with no transform
    wout,Xout,yout = linear_regression(len(t_setout),t_setout,lda1)
    print 'For 1 vs 5 Eout = %s'%(compute_Ein(wlin,Xout,yout))
    # in sample with transform
    t_set_trans = transform_t_set(t_set)
    wtrans,Xtrans,ytrans = linear_regression(len(t_set_trans),t_set_trans,lda1)
    # out of sample with transform        
    t_setout = transform_t_set(t_setout)
    wt,xt,yt = linear_regression(len(t_setout),t_setout,lda1)
    print 'For 1 vs 5 with transformation Ein = %s'%(compute_Ein(wtrans,Xtrans,ytrans))
    print 'For 1 vs 5 with transformation Eout = %s'%(compute_Ein(wtrans,xt,yt))   
    print '--------------------------------------------------'
    print 'lambda is: %s'%(lda2)
    # in sample with no transform
    wlin2,X02,y02 = linear_regression(len(t_set),t_set,lda2)
    print 'For 1 vs 5 Ein = %s'%(compute_Ein(wlin2,X02,y02))
    # out of sample with no transform
    wout2,Xout2,yout2 = linear_regression(len(t_setout2),t_setout2,lda2)
    print 'For 1 vs 5 Eout = %s'%(compute_Ein(wlin2,Xout2,yout2))
    # in sample with transform
    t_set_trans2 = transform_t_set(t_set)
    wtrans2,Xtrans2,ytrans2 = linear_regression(len(t_set_trans2),t_set_trans2,lda2)
    # out of sample with transform        
    t_setout2 = transform_t_set(t_setout2)
    wt2,xt2,yt2 = linear_regression(len(t_setout2),t_setout2,lda2)
    print 'For 1 vs 5 with transformation Ein = %s'%(compute_Ein(wtrans2,Xtrans2,ytrans2))
    print 'For 1 vs 5 with transformation Eout = %s'%(compute_Ein(wtrans2,xt2,yt2))


def tests():
    dFeaturesTrain = getDataFeatures('features.train')
    dFeaturesTest = getDataFeatures('features.test')
    print '-1-'
    print '-2-'
    print '-3-'
    print '-4-'
    #run_poly_kernel(dFeaturesTrain,dFeaturesTest)
    print '-5-'
    print '-6-'
    #run_1vs5_q2_q5(dFeaturesTrain,dFeaturesTest)
    print '-7-'
    print '-8-'
    #run_cross_validation(dFeaturesTrain,dFeaturesTest)
    print '-9-'
    print '-10-'
    #run_rbf_kernel(dFeaturesTrain,dFeaturesTest)
    print '-Bonus-'
    #run_reg_linear_reg_one_vs_all(dFeaturesTrain,dFeaturesTest)
    #run_reg_linear_reg_one_vs_one(dFeaturesTrain,dFeaturesTest)

