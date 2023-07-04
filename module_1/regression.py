import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    f_x = np.dot(w, x) + b
    
    for i in range(0, m):
        
        total_cost += (f_x[i] - y[i])**2
    
    total_cost = total_cost/(2*m)
    
    ### END CODE HERE ### 

    return total_cost