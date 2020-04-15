import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #First define some constant dimension numbers:
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  
  #Iterating over the length of minibatch.
  for n in range(N):
        #Calculate the first part of the loss
        exp_dot = np.dot(X[n, :], W)               #1*C
        #To make the exp vale stable. set logC = -max(fj)
        exp_stable = np.exp(exp_dot - np.max(exp_dot))
        #Calculate the probability 
        prob = exp_stable/np.sum(exp_stable)       #1*C
        loss = loss + (-np.log(prob[y[n]]))                #Scalar
        
        #Calculate the gradient, here we  have dLi/dw(j,i) = Xj*(Pi - 1) if i == label, = Xj*Pi
        #Iterating the feature vector
        for j in range(D):
              #Iterating over number of class
              for i in range(C):
                    #If i equal to actual label.
                    if i == y[n]:
                      dW[j, i] += X[n, j]*(prob[i] - 1)
                    else:
                      dW[j, i] += X[n, j]*prob[i]
        
  #Adding the regularization term, here we define L = 1/N*L1 + reg*<W, W>^2
  loss = loss/N
  dW   = dW/N
  loss = loss + reg*np.sum(W*W)
  dW   = dW + 2*reg*W      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #First define some constant dimension numbers:
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  
  #Then calculate the loss.
  exp_dot = np.dot(X, W)   #N*C
  #To make the exp vale stable. set logC = -max(fj)
  exp_stable = np.exp(exp_dot - np.max(exp_dot, axis=1).reshape(N, 1)) #N*C
  #Calculate the probability
  prob = exp_stable/np.sum(exp_stable, axis=1).reshape(N, 1) #N*C
  #loss with regularization
  loss = np.sum(-np.log(prob[range(N), list(y)]))/N + reg*np.sum(W*W)
  
  #Calculate the gradient
  #First minus 1 from prob matrix, according the gradient formulation
  prob[range(N), list(y)] -= 1
  dW = (X.T).dot(prob)/N + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

