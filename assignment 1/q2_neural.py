import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # print "data shape", data.shape
    # print b1.shape
    # print b2.shape
    x_trans = np.append(np.ones((data.shape[0],1)),data,axis=1)
    # print "x_trans", x_trans.shape
    w1_trans = np.append(b1,W1,axis=0)
    # print "w1_trans", w1_trans.shape
    a1 = np.dot(data, W1) + b1
    # print "a1",a1.shape
    h = sigmoid(a1)
    # print "h",h.shape
    h_trans = np.append(np.ones((h.shape[0],1)), h,axis=1)
    # print "h_trans", h_trans.shape
    w2_trnas = np.append(b2,W2,axis=0)
    # print "w2_trnas",w2_trnas.shape
    a2 = np.dot(h_trans, w2_trnas)
    # print "a2",a2.shape
    y = softmax(a2)
    cost = -np.sum(np.log(y)*labels)
    print "cost", cost

    ### END YOUR CODE
    ### YOUR CODE HERE: backward propagation

    d1 = y - labels
    # print "d1", d1.shape
    # print "d1", d1
    gradW2 = h.T.dot(d1)
    gradb2 = d1.sum(axis=0)
    # print "gradb2",gradb2.shape
    # print "gradW2", gradW2
    d2 = np.dot(d1, W2.T)
    # print "d2", d2.shape
    d3 = d2 * sigmoid_grad(h)
    gradW1 = data.T.dot(d3)
    gradb1 = np.sum(d3, axis=0)
    # print "gradW1.shape", gradW1.shape
    # print "gradb1.shaoe", gradb1.shape
    
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    # print "grad", grad
    # print "cost", cost
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()