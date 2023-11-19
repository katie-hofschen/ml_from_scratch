import numpy as np

"""
Non-linear activations functions allow non-linear learning -> more complexity.

A few rules of thumb:
- ReLU activation function should only be used in the hidden layers.
- Sigmoid/Logistic and Tanh functions should not be used in hidden layers as they make the model more susceptible to problems during training 
  (due to vanishing gradients).
- Swish function is used in neural networks having a depth greater than 40 layers.
- Regression - Linear Activation Function
- Binary Classification — Sigmoid
- Multiclass Classification — Softmax
- Multilabel Classification — Sigmoid
- For hidden layers of CNN: ReLU activation function.
- For hidden layers of Recurrent Neural Network: Tanh and/or Sigmoid 
"""

def sigmoid(x):
    """
    Sigmoid [0,1]

    Commonly used for models where we have to predict the probability as an output due to the range [0,1].
    The function is differentiable and provides a smooth gradient, i.e., preventing jumps in output values. 
    Disadvantages:
    - Vanishing gradient problem.
    - The output is not symmetric around zero. This makes the training of the neural network more difficult and unstable.
    """
    return 1 / (1 + np.exp(-x))

def reLU(x):
    """
    Rectified linear unit

    Advantages:
    - Since only a certain number of neurons are activated, the ReLU function is far more computationally efficient.
    - ReLU accelerates the convergence of gradient descent towards the global minimum of the loss function due to its linear, non-saturating property.

    Disadvantages:
    - Dying ReLU problem (the weights and biases for some neurons are not updated. This can create dead neurons which never get activated. 
                          Negative input values become zero immediately, which decreases the model's ability to fit or train as well.)
    """
    return np.max(x, 0)

def softmax(X):
    """
    the SoftMax function returns the probability of each class.
    most commonly used as an activation function for the last layer of the neural network in the case of multi-class classification.
    """
    sumX = np.sum([np.exp(x) for x in X])
    return [np.exp(x) / sumX for x in X]


def tanh(x):
    """
    hyperbolic tangent [-1,1]

    Advantages:
    - Output is Zero centered -> easily map the output values as strongly negative, neutral, or strongly positive.
    - Usually used in hidden layers of a neural network; Mean for the hidden layer ~ 0. Makes learning for the next layer much easier.
    Disadvantage:
    - problem of vanishing gradients
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_reLU(x):
    """
    Leaky ReLU is an improved version of ReLU function to solve the Dying ReLU problem as it has a small positive slope in the negative area.
    
    Advantages:
    - same as reLU
    Disadvantages:
    - The predictions may not be consistent for negative input values. 
    - The gradient for negative values is a small value that makes the learning of model parameters time-consuming.
    """
    return np.max(0.1*x, x)

def parametric_reLU(x, p):
    """
    Another variant of ReLU that aims to solve the problem of gradient's becoming zero for the left half of the axis

    TODO double check this.
    """
    return np.max(p*x , x)

def elu(x , a):
    """
    uses a log curve to modify the negative part of the ReLU slope

    Advantages:
    - becomes smooth slowly until its output equal to -a whereas RELU sharply smoothes.
    - Avoids dead ReLU problem by introducing log curve for negative values of input. 
    Disadvantages:
    - It increases the computational time because of the exponential operation included
    - No learning of the a value takes place
    - Exploding gradient problem
    """
    if x >= 0:
        return x
    else:
        return a * (np.exp(x)-1)


def swish(x):
    """
    Proposed in 2017 by Ramachandran et.al.
    self-gated activation function developed by researchers at Google. 
    Swish consistently matches or outperforms ReLU activation function on deep networks applied to various challenging domains
    such as image classification, machine translation etc. 

    advantages of the Swish activation function over ReLU:
    - Smooth function -> doesn't abruptly change direction like ReLU does near x = 0.   
    - Small negative values may be relevant for capturing patterns underlying the data. 
      Large negative values are zeroed out for reasons of sparsity making it a win-win situation.
    - Is non-monotonous -> enhances the expression of input data and weight to be learnt.

    Disadvantages:
    - Computationally expensive (especially if called for each neuron in a deep NN)
    """
    sw = x * sigmoid(x)
    return sw


def hard_swish(x):
    hsw = x * (parametric_reLU(x+3, 6) / 6)


def gelu(x):
    """
    Gaussian Elu

    Compatible with BERT, ROBERTa, ALBERT, and other top NLP models. 
    This activation function is motivated by combining properties from dropout, zoneout, and ReLUs.

    We multiply the neuron input x by 
    m ~ Bernoulli(Φ(x)), where Φ(x) = P(X ≤x), X ~ N (0, 1) is the cumulative distribution function of the standard normal dist. 
    
    GELU nonlinearity is better than ReLU and ELU activations and finds performance improvements across 
    all tasks in domains of computer vision, natural language processing, and speech recognition.
    """
    part = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x,3))
    return 0.5 * x ( 1 + tanh(part))

def selu(x, a, l):
    """
    Scaled Exponential Linear Unit (SELU)
    SELU was defined in self-normalizing networks and takes care of internal normalization which means each layer preserves the mean and variance from the previous layers. SELU enables this normalization by adjusting the mean and variance. 
    SELU has both positive and negative values to shift the mean, which was impossible for ReLU activation function as it cannot output negative values. 
    Gradients can be used to adjust the variance. The activation function needs a region with a gradient larger than one to increase it.

    Main advantage of SELU over ReLU:
    Internal normalization is faster than external normalization, which means the network converges faster.

    TODO relatively new, check current research on performance.
    TODO check if correct
    """
    if x < 0:
        l * a (np.exp(x) - 1)
    else:
        x