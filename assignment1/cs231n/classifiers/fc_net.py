from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["b1"] = np.zeros((hidden_dim,))
        self.params["b2"] = np.zeros((num_classes,))
        self.params["W1"] = np.random.normal(
            scale=weight_scale, size=(input_dim, hidden_dim)
        )
        self.params["W2"] = np.random.normal(
            scale=weight_scale, size=(hidden_dim, num_classes)
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        train_num = X.shape[0]
        X_flatten = X.reshape(train_num,-1)
        scores_hidden = X_flatten.dot(self.params["W1"]) + self.params["b1"]
        scores_activated = np.where(scores_hidden < 0, 0, scores_hidden)
        scores = scores_activated.dot(self.params["W2"]) + self.params["b2"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        scores_shifted = scores - np.max(scores, axis=1)[:, np.newaxis]
        e_scores = np.exp(scores_shifted)
        prob = e_scores / np.sum(e_scores, axis=1)[:, np.newaxis]
        loss = np.mean(-np.log(prob[np.arange(train_num), y]))
        loss += (
            self.reg
            * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
            / 2
        )

        dL_dS = prob
        dL_dS[np.arange(train_num), y] -= 1
        dL_dS /= train_num
        dL_dW2 = (scores_activated.T).dot(dL_dS) +  self.reg * self.params["W2"]
        dL_db2 = np.sum(dL_dS, axis=0)
        dL_dsa = dL_dS.dot(self.params["W2"].T)

        dsa_dsh = np.where(scores_activated > 0, 1, 0)
        dL_dsh = dL_dsa * dsa_dsh

        dL_dW1 = X_flatten.T.dot(dL_dsh)+ self.reg * self.params["W1"]
        dL_dX = dL_dsh.dot(self.params["W1"].T)
        dL_db1 = np.sum(dL_dsh, axis=0)

        grads["W2"] = dL_dW2
        grads["b2"] = dL_db2
        grads["W1"] = dL_dW1
        grads["b1"] = dL_db1
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
