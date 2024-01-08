from builtins import range
import numpy as np
from random import shuffle

# from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    train_num, dim = X.shape
    class_num = W.shape[1]

    for i in range(train_num):
        score = X[i].dot(W)
        score = score - np.max(score)
        e_score = np.exp(score)
        prob = e_score / np.sum(e_score)
        loss += -np.log(prob[y[i]])

        for j in range(class_num):
            if j == y[i]:
                dW[:, j] += X[i] * (-1 + prob[j])
            else:
                dW[:, j] += X[i] * prob[j]

    loss /= train_num
    loss += reg * np.sum(W * W)
    dW /= train_num
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    train_num, dim = X.shape
    class_num = W.shape[1]
    coeff = np.zeros((train_num, class_num))

    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1)[:, np.newaxis]
    e_scores = np.exp(scores)
    prob = e_scores / np.sum(e_scores, axis=1)[:, np.newaxis]
    correct_prob = prob[np.arange(train_num), y]
    loss = -np.log(correct_prob)
    loss = np.mean(loss) + reg * np.sum(W * W)

    coeff = prob
    coeff[np.arange(train_num), y] -= 1

    dW = X.T.dot(coeff)
    dW = dW / train_num + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
