import math

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def eggholder(args):
    ''' http://www.sfu.ca/~ssurjano/egg.html '''
    x = args["x"]
    return (-(x[1]+47)*math.sin(math.sqrt(abs(x[1]+(x[0]/2.0)+47)))-x[0]*math.sin(math.sqrt(abs(x[0]-(x[1]+47)))), None)

def michal(args):
    ''' http://www.sfu.ca/~ssurjano/michal.html '''
    x = args["x"]
    sum = 0
    for i in range(len(x)):
        sum-=math.sin(x[i])*math.sin(((i+1)*x[i]**2)/math.pi)**20
    return (sum, None)

def paraboloid(args):
    x = args["x"]
    return (x[0]**2 +x[1]**2, None)

def shubert(args):
    ''' http://www.sfu.ca/~ssurjano/shubert.html '''
    x = args["x"]
    sum_0 = 0
    sum_1 = 0
    for i in range(1,6):
        sum_0+=i*math.cos((i+1)*x[0]+i)
        sum_1+=i*math.cos((i+1)*x[1]+i)
    return (sum_0*sum_1, None)

def rastrigin(args):
    ''' https://www.sfu.ca/~ssurjano/rastr.html '''
    x = args["x"]
    _sum = 10*len(x)
    for i in range(len(x)):
        _sum+=x[i]**2-10*math.cos(2*math.pi*x[i])
    return (_sum, None)

def get_standard_funcs():
    funcs = [ eggholder, michal, paraboloid, shubert, rastrigin ]
    return [ (func.__name__, func) for func in funcs ]