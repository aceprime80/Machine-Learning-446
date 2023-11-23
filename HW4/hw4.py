import torch
import hw4_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    w = torch.zeros((X.shape[1]+1), 1)
    Xn = torch.ones((X.shape[0], X.shape[1]+1))
    Xn[:,1:] = X 
    for x in range(num_iter):
        Y_pred = Xn @ w

        error = Y_pred - Y

        # Compute the gradient
        gradient = Xn.t() @ error
        #print(w)
        # Update the parameters
        w -= lrate * gradient / X.shape[0]
        
    return w

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    Xn = torch.ones((X.shape[0], X.shape[1]+1))
    Xn[:,1:] = X 
    # print(torch.pinverse(Xn).shape)
    # print(Y.shape)
    return torch.pinverse(Xn) @ Y

    

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X,Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    #print(w)
    Y_pred = X*w[1] + w[0]
    # MAKE IT PRETTY
    plt.plot(X, Y_pred, c="orange")
    plt.scatter(X,Y)
    plt.title("Linear Regression Plot")
    plt.show()
    return plt.gcf()

#plot_linear()
# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    d = X.shape[1]
    Xn = torch.ones((X.shape[0],1+d+d*(d+1) // 2))
    Xn[:,1:1+d] = X
    p = 1+d
    j = 0
    for i in range(d):
        for j in range(i,d):
            Xn[:, p] = X[:,i] * X[:,j]
            p+=1
    w = torch.zeros((1+d+d*(d+1) // 2, 1))
    for x in range(num_iter):
        Y_pred = Xn @ w

        error = Y_pred - Y

        # Compute the gradient
        gradient = Xn.t() @ error
        #print(w)
        # Update the parameters
        w -= lrate * gradient / X.shape[0]
        
    return w
    

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    d = X.shape[1]
    Xn = torch.ones((X.shape[0],1+d+d*(d+1) // 2))
    Xn[:,1:1+d] = X
    p = 1+d
    for i in range(d):
        for j in range(i, d):
            Xn[:, p] = X[:,i] * X[:,j]
            p+=1
    
    return torch.pinverse(Xn) @ Y

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X,Y = utils.load_reg_data()
    w = poly_normal(X,Y)
    #print(w)
    d = X.shape[1]
    Xn = torch.ones((X.shape[0],1+d+d*(d+1) // 2))
    Xn[:,1:1+d] = X
    p = 1+d
    for i in range(d):
        for j in range(i, d):
            Xn[:, p] = X[:,i] * X[:,j]
            p+=1
    Y_pred = Xn @ w
    # MAKE IT PRETTY
    plt.plot(X, Y_pred, c="orange")
    plt.scatter(X,Y)
    plt.title("Poly-regression Plot")
    plt.show()
    return plt.gcf()

#plot_poly()
def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    X,Y = utils.load_xor_data()
    w_poly = poly_normal(X,Y)
    def poly_pred(X):
        d = X.shape[1]
        X_poly = torch.ones((X.shape[0],1+d+d*(d+1) // 2))
        X_poly[:,1:1+d] = X
        p = 1+d
        for i in range(d):
            for j in range(i, d):
                X_poly[:, p] = X[:,i] * X[:,j]
                p+=1
        
        Y_poly_pred = X_poly @ w_poly 
        return Y_poly_pred
    
    w_lin = linear_normal(X,Y)
    def lin_pred(X):
        X_lin = torch.ones((X.shape[0], X.shape[1]+1))
        X_lin[:,1:] = X 
        Y_lin_pred = X_lin @ w_lin
        return Y_lin_pred
    
    utils.contour_plot(-1,1,-1,1,lin_pred,ngrid=66)
    utils.contour_plot(-1,1,-1,1,poly_pred,ngrid=66)
    return lin_pred(X), poly_pred(X)
#print(poly_xor())


