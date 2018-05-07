from clustering import Clustering
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

np.seterr(all='raise')
cl = Clustering()
#

#load data and correct labels (y)
#data = np.genfromtxt("iris.csv", delimiter=',')
data_ori = np.genfromtxt("x_train_data", delimiter=' ')
data = StandardScaler().fit_transform(data_ori)
#put labels in y
#y = np.zeros(120)
#y[30:60] = 1
#y[60:90] = 2
#y[90:120] = 3
'''
y = np.zeros(150)
y[50:100] = 1
y[100:150] = 2
'''
#
#y=y.astype(int)

#data,y=load_iris(True)
# standardize data
#data = cl.my_math.standardize(data)
#apply one algorithm at a time
####################################################################
'''
print('PAM')
start = time.time()
[u, centroids, ite, dist_tmp] = cl.pam(data, 3, replicates=10)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################
print('Build PAM')
start = time.time()
[u, medoids, ite, dist_tmp] = cl.build_pam(data, 3)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################
print('Minkowski Weighted PAM')
start = time.time()
[u, medoids, weights, ite, dist_tmp] = cl.mwpam(data, 3, 1.1, False, 10)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################
print('Minkowski Weighted PAM (Initialized with Minkowski Build)')
start = time.time()
[u, medoids, weights, ite, dist_tmp] = cl.mwpam(data, 3, 1.1)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################

print('K-Means')
start = time.time()
[u, centroids, ite, dist_tmp] = cl.k_means(data, 4, replicates=10)
print('Time elapsed: ', time.time()-start)
#print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])

####################################################################
print('iK-Means')
start = time.time()
[u, centroids, ite, dist_tmp, init_centroids] = cl.ik_means(data, 3)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################
'''
print('WK-Means')
start = time.time()
[u, centroids, weights, ite, dist_tmp] = cl.wk_means(data, 40, 5, replicates=1, max_ite=10, init_weights_method="random")
print weights,centroids
np.savetxt("centroids", centroids, fmt="%.4f")
np.savetxt("weights", weights, fmt="%.4f")
print('Time elapsed: ', time.time()-start)
#print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################
'''
print('MWK-Means')
start = time.time()
[u, centroids, weights, ite, dist_tmp] = cl.mwk_means(data, 40, 1.1, replicates=1, max_ite=10, init_weights_method="fixed")
np.savetxt("centroids", centroids, fmt="%.4f")
np.savetxt("weights", weights, fmt="%.4f")
print('Time elapsed: ',time.time()-start)
#print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])
####################################################################

print('iMWK-Means')
start = time.time()
[u, centroids, weights, ite, dist_tmp] = cl.imwk_means(data, 1.1, 3)
print('Time elapsed: ', time.time()-start)
print('Accuracy: ', cl.my_math.compare_categorical_vectors(u, y)[0])

####################################################################
print('proclus')
start = time.time()
R = 1 # toggle run proclus
RS = 1 # toggle use random seed

if R: # run proclus
    rseed = 902884
    if RS:
        rseed = np.random.randint(low = 0, high = 1239831)
        #print X
        #print sup
    print "Using seed %d" % rseed
    M, D, A = prc.proclus(X, k = 3, l = 3, seed = rseed)
    print A
    print "Accuracy: %.4f" % prc.computeBasicAccuracy(A, target)
print('Time elapsed: ', time.time()-start)
'''
