import hw5_utils as utils
import numpy as np
import torch
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n,d = x_train.shape
    alpha = torch.zeros(n, requires_grad=True)
    
    for iters in range(num_iters):
        loss = 0.0
        
        for i in range(n):
            for j in range(n):
                loss += alpha[i]*alpha[j]*y_train[i]*y_train[j]*kernel(x_train[i],x_train[j])
        loss *= 0.5
        loss -= alpha.sum()
        loss.backward()
        with torch.no_grad():
            alpha -= lr*alpha.grad
            if c is None:
                alpha = alpha.clamp_(min = 0)
            else:
                alpha = alpha.clamp_(min = 0, max = c)
        alpha.grad.zero_()
    return alpha.detach()
#X,Y = utils.xor_data()
#print(svm_solver(X,Y,0.01, 100))

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    n = x_train.shape[0]
    m = x_test.shape[0]
    pred = torch.zeros(m)
    for i in range(m):
        for j in range(n):
            pred[i] += alpha[j]*y_train[j]*kernel(x_train[j],x_test[i])
    return pred
    
        

def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n,d = X.shape
    X = torch.cat((torch.ones((n,1)), X), 1)
    w = torch.zeros((d+1,1))
    #print(X)
    for i in range(num_iter):
        w -= (lrate/n)*((-Y*X/(1+torch.exp((Y*X)@w))).sum(axis=0)).reshape((d+1,1))
        # print("||||||||||||||||")
        # print(sigmoid((Y*X)@w))
    return w
#X,Y = utils.load_logistic_data()
#print(logistic(X,Y))
def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    def linear_normal(X, Y):
        X = torch.cat((torch.ones((X.shape[0],1)), X), 1)
        # print(torch.pinverse(Xn).shape)
        # print(Y.shape)
        return torch.pinverse(X) @ Y
    
    X,Y = utils.load_logistic_data()
    w_lin = linear_normal(X,Y)
    w_log = logistic(X,Y, num_iter=1000000)
    #print(w_lin)
    temp = torch.linspace(-5,5,10)
    Y_lin = -(temp*w_lin[1] + w_lin[0])/w_lin[2]
    Y_log = -(temp*w_log[1] + w_log[0])/w_log[2]
    # MAKE IT PRETTY
    plt.plot(temp, Y_lin, c="orange")
    plt.plot(temp, Y_log, c="blue")
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.title("Linear (orange) vs Logistic (blue) Classification")
    plt.show()
    return plt.gcf()

logistic_vs_ols()


X,Y = utils.xor_data()
alpha_poly = svm_solver(X,Y,0.1,10000,kernel=utils.poly(degree=2))
pred_poly = lambda test: svm_predictor(alpha_poly,X,Y,test,utils.poly(degree=2))
utils.svm_contour(pred_poly)

alpha_rbf1 = svm_solver(X,Y,0.1,10000,kernel=utils.rbf(1))
pred_rbf1 = lambda test: svm_predictor(alpha_rbf1,X,Y,test,utils.rbf(1))
utils.svm_contour(pred_rbf1)

alpha_rbf2 = svm_solver(X,Y,0.1,10000,kernel=utils.rbf(2))
pred_rbf2 = lambda test: svm_predictor(alpha_rbf2,X,Y,test,utils.rbf(2))
utils.svm_contour(pred_rbf2)

alpha_rbf4 = svm_solver(X,Y,0.1,10000,kernel=utils.rbf(4))
pred_rbf4 = lambda test: svm_predictor(alpha_rbf4,X,Y,test,utils.rbf(4))
utils.svm_contour(pred_rbf4)