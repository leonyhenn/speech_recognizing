from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import os, fnmatch
import random
import math

dataDir = '/u/cs401/A3/data/'
# dataDir = '../data/'


class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))

def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    #checked
    
    sigma = myTheta.Sigma[m]
    
    mu = myTheta.mu[m]
    d = x.shape[1]
    
    pc_1 = np.sum(np.true_divide(np.square(mu),2 * sigma))
    pc_2 = (d / 2) * math.log(2 * math.pi)
    pc_3 = (1 / 2) * np.log(np.prod(sigma))

    front = -(np.sum((1/2)*(np.true_divide(np.square(x),sigma)) - np.true_divide(np.multiply(mu,x),sigma),axis=1)) 
    result = np.subtract(front,(pc_1 + pc_2 + pc_3))

    return result

def log_p_m_x_logBs(logBs,myTheta):

    top = np.add(logBs, np.log(myTheta.omega))
    bottom = scipy.misc.logsumexp(logBs, b=myTheta.omega, axis=0)
    result = top - bottom

    return result

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    #checked
    top = np.log(myTheta.omega[m]) + log_b_m_x(m, x, myTheta)
    bottom = []
    for k in range(myTheta.omega.shape[0]):
        bottom.append(np.log(myTheta.omega[k]) + log_b_m_x(k, x, myTheta))
    bottom = scipy.misc.logsumexp(bottom)
    result = top - bottom    
    return result

def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    #checked
    result = np.sum(scipy.misc.logsumexp(log_Bs, b=myTheta.omega, axis=0))
    return result
    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # initialize theta
    myTheta = theta( speaker, M, X.shape[1] )
    
    # init omega (8x1), Initialize omega randomly, with each 0<=omega<=1, sum to 1
    myTheta.omega = np.random.dirichlet(np.ones(M),size=1).reshape(M,1)
        
    # init mu (8x13), Initialize each mu to a random vector from the data.
    randomIndexList = random.sample(range(0, X.shape[0]), myTheta.mu.shape[0])
    for i in range(myTheta.mu.shape[0]):
        myTheta.mu[i] = X[randomIndexList[i]]
    
    # init Sigma (8x13), Initialize Sigma set to 1/M 
    myTheta.Sigma.fill(1/M)

    # i := 0
    iteration = 0
    
    # prev_L = -inf
    prev_L = float("-inf")
    
    # improvement = inf
    improvement = float("inf")

    T = X.shape[0]

    temp_log_b = np.zeros((M,T))

    # while i <= maxiter and improvement >= epsilon:
    while iteration <= maxIter and improvement >= epsilon:

    #   ComputeIntermediateResults
        for m in range(M):
            temp_log_b[m,] = log_b_m_x(m, X, myTheta)
            
        temp_log_p = log_p_m_x_logBs(temp_log_b, myTheta)
        
    #   L := ComputeLikelihood(X,myTheta)
        L = logLik(temp_log_b, myTheta)

    #   myTheta := UpdateParameters(myTheta, X, L)
        p_m_x_theta = np.exp(temp_log_p)

        myTheta.omega = np.true_divide(np.sum(p_m_x_theta,axis=1).reshape((M,1)),T)
        myTheta.mu = np.true_divide(np.dot(p_m_x_theta,X),np.sum(p_m_x_theta,axis=1).reshape((M,1)))
        myTheta.Sigma = np.subtract(np.true_divide(np.dot(p_m_x_theta,np.square(X)),np.sum(p_m_x_theta,axis=1).reshape((M,1))),np.square(myTheta.mu))

        
    #   improvement := L - prev_L
        improvement = L - prev_L
    
    #   prev_L = L
        prev_L = L
        
    #   i := i+1
        iteration = iteration + 1
    
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    M = models[0].omega.shape[0]
    T = mfcc.shape[0]
    AllLogBs = []
    for i in range(len(models)):
        temp_log_b = np.zeros((M,T))
        for m in range(M):
            temp_log_b[m,] = log_b_m_x(m, mfcc, models[i])
        L = logLik(temp_log_b, models[i])
        AllLogBs.append(L)
    max_value = max(AllLogBs)

    bestModel = AllLogBs.index(max_value)
    
    print(bestModel,correctID)
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

