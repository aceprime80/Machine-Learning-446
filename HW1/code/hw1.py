import torch
import hw1_utils
import numpy as np

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """
    

    if X is None:
        X, init_c = hw1_utils.load_data()
    
    c = init_c #start are centroids at the initial positions

    # first loop is to represent one full update
    # currently does not stop once there are no more changes but can be easily added
    for i in range(n_iters):
        # r is the r in our equation representation of a step. 
        # a matrix  where r_ik is 1 when the kth centroid corresponds to the ith point.
        r = torch.zeros((X.shape[1], init_c.shape[1]))
        # This loop is the first step in an update in which a we find each points distance
        # to each centroid and r_ik = 1 where k is the closest centroid to our point
        # and i is our point(in the code its j).
        for j in range(X.shape[1]):
            #finding the distance from point to centroid
            distances = torch.zeros(2)
            for w in range(c.shape[1]):
                diff = (X[:,j]-c[:,w])
                distances[w] = diff@diff
            #print(distances)
            r[j, torch.argmin(distances)] = 1  #assigning r_ik
        
        
        numofpoints = torch.sum(r,0)
        points0 = torch.zeros((2, int(numofpoints[0].item())))
        points1 = torch.zeros((2, int(numofpoints[1].item())))
        c1 = torch.zeros((2,1))
        c1[:,0] = c[:,0]
        c2 = torch.zeros((2,1))
        c2[:,0] = c[:,1]
        for j in range(c.shape[1]):
            count = 0
            c[:,j] = torch.zeros(2)
            for k in range(X.shape[1]):
                
                if r[k,j] == 1:
                    if (j == 0):
                        points0[:,count] = X[:,k]
                    else:
                        points1[:,count] = X[:,k]
                    count += 1
                    c[:,j] += X[:,k]
            c[:,j] /= count
        
        print(c)
        hw1_utils.vis_cluster(c1,points0,c2, points1)

    
    return c


print(k_means())
