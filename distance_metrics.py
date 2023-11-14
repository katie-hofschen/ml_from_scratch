import numpy as np

"""
A file to collect all distance measures in one place.
"""

def euclidian_dist(v1, v2):
    """
    if a triangle is drawn with the 2 points as corners then the euclidian distance is the hypothenuse
    """
    return np.sqrt(np.sum([np.square(s - x) for x,s in zip(v1, v2)]))

def manhattan_dist(v1, v2):
    """
    this distance is equal to moving across a grid of squares from point a to point b (no diagonal)
    """
    return np.sum([abs(s - x) for x,s in zip(v1,v2)])

def minkowski_dist(v1, v2, p=3):
    """
    if p = 1 then it is equivalent to the manhattan distance
    if p = 2 then it is equivalent to the euclidian distance
    """
    return np.power(np.sum([np.power((s - x), p) for x,s in zip(v1, v2)]), 1/p)


def chebyshev_dist(v1, v2):
    """
    choses the maximal difference between 2 variables of the 2 vectors 
    visually it choses the side of the triangle that is longer and not the hypothenuse
    """
    return np.max([abs(s - x) for x,s in zip(v1,v2)])

def cosine_distance(v1, v2):
    """
    Distance metric that uses the angle of the vectors to determine whether they point in a similar direction
    norm(v1) == np.sqrt(np.dot(v1,v1)) using (from numpy.linalg import norm)
    """
    return 1 - ( np.dot(v1, v2) / ( np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)) ) )

def jaccard_distance(v1, v2):
    """ 
    Distance metric that works well for categorical variables
    may be a poor metric if there are no positives for some samples or classes
    is undefined if there are no true or predicted labels
    """
    anb = len(set(v1).intersection(set(v2))) 
    aub = len(set(v1).union(set(v2)))
    return 1 - (anb / aub)

def hamming_distance(v1, v2):
    """
    Distance metric that returns the proportion of the number of substitutions needed to change one string into another
    also works well for categorical variables or binary strings such as one hot encoding
    """
    if len(v1) != len(v2):
        raise ValueError("For the Hamming distance the 2 vectors must have the same length. Please check this.")
    return np.sum([s != x for x,s in zip(v1,v2)])/ len(v1)