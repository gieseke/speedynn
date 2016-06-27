import numpy
import random

class Gaussians:
    
    # Default parameters
    parameters = {
    'nums': [100,100],
    'class_ids' : [-1,1],
    'centers': [[-5, -5],[5,5]],
    'variances': [[1.0,1.0],[1.0,1.0]],
    'filename': None,
    'seed':0    
    }
    
    def __init__(self,  ** kw):
        self.setParameters( ** kw)
        random.seed(self.__seed)
        
    def setParameters(self,  ** kw):
        for attr, val in kw.items():
            self.parameters[attr] = val
        self.__nums = self.parameters['nums']
        self.__class_ids = self.parameters['class_ids']
        self.__centers = self.parameters['centers']
        self.__variances = self.parameters['variances']
        self.__filename = self.parameters['filename']            
        self.__dim = len(self.__centers[0])
        assert len(self.__nums) == len(self.__centers)
        assert len(self.__nums) == len(self.__variances)
        self.__seed = int(self.parameters['seed'])        
        
    def generate(self):
        L = []
        X = []
        # for each cluster
        for i in xrange(len(self.__nums)):
            num = self.__nums[i]
            class_id = self.__class_ids[i]
            center = self.__centers[i]
            variance = self.__variances[i]
            for j in xrange(num):
                x = []
                for k in xrange(self.__dim):
                    x.append(random.gauss(center[k], variance[k]))
                #x = numpy.array(x)
                X.append(x)
                L.append(class_id)
            
        if self.__filename != None:
            ofile = open(self.__filename, 'w')
            for i in xrange(len(L)):
                ofile.write('%g ' % L[i])
                for j in xrange(len(X[i])):
                    ofile.write(str(j + 1) + ':%+12.5e ' % X[i][j])
                ofile.write('\n')
            ofile.close()                        
        return X,L


def Gaussian(n=10, dim=2, seed=0):
    N = n/2
    dist_clusters_on_xaxis = 5.0
    dim_Change = 1
    nums = [N/2,N/2]
    class_ids = [-1,1]
    centers = [[0.0 for d in xrange(dim)] for elt in xrange(len(nums))]
    # For each cluster: Different position on x-axis
    for i in xrange(len(nums)):
        for j in range(dim_Change):
            centers[i][j] = (-(0.5*dist_clusters_on_xaxis)+i*2*(0.5*dist_clusters_on_xaxis))
    variances = [[1.0 for d in xrange(dim)] for elt in xrange(len(nums))]
    # Change center and variance for the first dimension (to create low-density area)
    for i in xrange(len(nums)):
        variances[i][0] = 1.0
    X,Y = Gaussians(nums=nums, class_ids=class_ids, centers=centers, variances=variances, filename=None, seed=seed).generate()
    X = numpy.array(X)
    Y = numpy.array(Y)
    return X[:N], Y[:N], X[N:], Y[N:]


    
def quasar_extended_features():
    feature_path = "../../data/quasarRedshiftRegression/quasars.features.rnd.npy"
    label_path = "../../data/quasarRedshiftRegression/quasars.labels.rnd.txt"
    X = numpy.load(feature_path)
    Y = numpy.loadtxt(label_path)
    return X,Y,None,None

