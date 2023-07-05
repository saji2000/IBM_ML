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

# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    
    cost = np.dot(w, x) + b
    
    for i in range(0, m):
        dj_dw += (cost[i] - y[i]) * x[i]
        dj_db += (cost[i] - y[i])
        
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db