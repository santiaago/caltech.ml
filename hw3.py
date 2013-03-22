from math import exp

EPSILON = .05
MAX_P_BOUND = 0.03

def pbound(m,n):
    return 2*m*exp(-2*(EPSILON**2)*n)

def leastnumberofexamples(m = 1):
    n = 0
    while pbound(m,n) > 0.03:
        n = n + 1
        if n > 5000: break
    return n
      
def tests():
    #Generalization error: 1-2-3
    print leastnumberofexamples(1)
    print leastnumberofexamples(10)
    print leastnumberofexamples(100)
    #Break point: 4
    #Growth function: 5
    #Fun with intervals: 6-7-8
    #Convex sets: The Triangle: 9
    #Non convex sets: concentric circles: 10

    
           
