import numpy as np

"""
A file to collect all metrics in one place
Expand over time.
"""

# Mean absolute error
def mae(y, pred):
    """
    The mean of the absolute error between y and the predicted.
    """
    mae = ( np.sum([abs(t-p) for t,p in zip(y, pred)]) / len(pred) )
    return np.round(mae, 4)

# Mean squared error
def mse(y, pred):
    """
    The mean squared error between y and the predicted.
    This metric penalizes larger errors more than MAE but due to the quadratic the error (unit^2) cannot be directly compared to the target (unit).
    This same property means that the metric can be more affected by outliers.
    """
    mse = ( np.sum([np.square(t-p) for t,p in zip(y, pred)]) / len(pred) )
    return np.round(mse, 4)

#Root mean squared error
def rmse(y, pred):
    """
    The root of the mean of the squared error between y and the predicted.
    While larger errors are still penalized, using the root allows a better interpretability/comparison to the target (unit).
    Also more susceptible to give outliers more importance.
    """
    rmse = np.sqrt( np.sum([np.square(t-p) for t,p in zip(y, pred)]) / len(pred) )
    return np.round(rmse, 4)

#Root mean squared logarithmic error
def rmsle(y, pred):
    """
    The root of the mean of the squared log(error) between y and the predicted.
    mainly used when predictions have large deviations (0 - millions) and we don't want to punish deviations in prediction as much as with MSE.
    """
    errors = [np.square( np.log1p(t) - np.log1p(p) ) for t,p in zip(y, pred)]
    rmsle= np.sqrt( np.sum(errors) / len(pred) )
    return np.round(rmsle, 4)

def accuracy(predicted, target):
    # ( TP + TN ) / ( all )
    return np.sum(predicted == target) / len(target)

def precision(predicted, target):
    # True positive / all predicted as positive
    # TP / ( TP + FP )
    pass

def recall(predicted, target):
    # True positive / all that should be predicted positive
    # TP / ( TP + FN )
    pass

def f_score(predicted, target, beta):
    # f_beta = (1 + np.squared(beta)) * ((Precision * recall) / (np.squared(beta) * precision + recall))
    pass

# can add True/False positive rate, Sensitivity, and more