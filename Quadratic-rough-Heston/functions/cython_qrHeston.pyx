#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
cimport numpy as np  
cimport cython
from libc.math cimport isnan
from tqdm import trange
from scipy.special import gamma


cdef extern from "math.h":
    double sqrt(double m)
    double exp(double m)
    double log(double m)
    double fabs(double m)
    
    

@cython.boundscheck(False)    # turn off bounds-checking for entire function
@cython.wraparound(False)     # turn off negative index wrapping for entire function


cpdef qrHeston_single_stockPaths(double S0, double T, double[:] time_grid, int nbPaths, 
                             double alpha, double Lambda, double a, double b,double c, double Z0):
        """
        Monte Carlo Simulation for the quadratic rough Heston model
        Input: 
            T: maturity
            nbPaths: number of paths
        
        Output:
            An array of Stock price paths at maturity T, with each row(axis=0) a simulation path 
            
        """
        ## choose the proper time grid for the maturity T
        cdef double dt     = time_grid[1] - time_grid[0]
        cdef double dtsqrt  = sqrt(dt)
        cdef int nbTimeSteps = int(T/dt)
        # Generate a Brownian Motion sequence
        cdef double[:,:] W = np.random.normal(0, dtsqrt, (nbPaths, nbTimeSteps))
        # Initialize the variables 
        cdef double[:,:] X = np.zeros((nbPaths, nbTimeSteps))
        cdef double[:,:] Z = np.zeros((nbPaths, nbTimeSteps))
        cdef double[:,:] V = np.zeros((nbPaths, nbTimeSteps))

        cdef double coef = Lambda/gamma(alpha)
        
        cdef double[:] ti
        
        for i in range(nbPaths):
            X[i, 0] = log(S0) 
            Z[i, 0] = Z0 
            V[i, 0] = a * (Z0 - b)**2 + c
            
        
        for i in trange(1, nbTimeSteps):
            
            for k in range(i):
                ti[k]   = np.power(time_grid[i] - time_grid[k], alpha - 1)
                Zi[:, k]   = np.asarray(Z[:, k])
                Vi[:, k]   = np.asarray(V[:, k])
                Wi[:, k]   = np.asarray(W[:, k])
            
            tmp = np.dot(dt*np.asarray(Zi) - sqrt(np.asarray(Vi))*np.asarray(Wi), np.asarray(ti))
            for j in range(nbPaths):
                Z[j, i] = Z0 - coef*tmp
                V[j, i] = a * (Z[j, i] - b)**2 + c
                X[j, i] = X[j, i-1] -0.5*V[j, i-1]*dt + sqrt(V[j, i-1])*W[j, i]
        
        return np.exp(np.asarray(X))
    
    
