# Base Python Imports
import csv

# Third Part Imports
import numpy as np


########################### Random Nanowire Networks ###########################

def create_nwn_end(N, X=1, Y=1, l0=1, seed=None):
    '''
    Generates a list of endpoints for a random network of wires using the "end-point" method
    '''

    if seed is not None:
        np.random.seed(seed)

    l = l0 #TODO: Make random variable with given pdf

    pos1 = np.concatenate((np.random.rand(N,1)*X, np.random.rand(N,1)*Y), axis=1)
    theta = np.random.rand(N)*2*np.pi
    pos2 = pos1 + (np.array([np.cos(theta), np.sin(theta)])*l).T

    return list(zip(pos1, pos2))

def create_nwn_mid(N, X=1, Y=1, l0=1, seed=None):
    '''
    Generates a list of endpoints for a random network of wires using the "mid-point" method
    '''
     
    if seed is not None:
        np.random.seed(seed)

    l = l0 #TODO: Make random variable with given pdf
    
    center = np.array([np.random.rand(N)*X, np.random.rand(N)*Y]).T
    theta = np.random.rand(N)*2*np.pi
    dp = (np.array([np.cos(theta), np.sin(theta)])*l/2).T
    pos1 = center + dp
    pos2 = center - dp

    return list(zip(pos1, pos2))

def csv_JDA():
    raise NotImplementedError

def csv_MNR():
    raise NotImplementedError

############################ Grid Nanowire Networks ############################

def csv_Grid(n, m):
    raise NotImplementedError

############################ Maze Nanowire Networks ############################

def csv_Maze_Wilson(n, m, start, end):
    raise NotImplementedError

