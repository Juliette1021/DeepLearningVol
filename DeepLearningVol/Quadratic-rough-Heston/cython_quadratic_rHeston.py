#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import math
from scipy.special import gamma
from scipy.optimize import bisect
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm.notebook import trange 
import time
import matplotlib.ticker as mtick

from functions.cython_qrHeston import qrHeston_single_stockPaths

### Price of the Call/Put option with the Black Scholes model

def BlackScholesCallPut(S, K, T, sigma, r=0.0, call_put=1):
    d1 = (np.log(S/K) + (r+.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put*(S*norm.cdf(call_put*d1) - K*np.exp (-r*T) * norm.cdf (call_put*d2))

#### Compute the implied volatility

def impliedVol(S, K, T, price, r=0.0, call_put=1):
    def smileMin(vol, *args):
        S, K, T, price, r, call_put = args
        return price - BlackScholesCallPut(S, K, T, vol, r, call_put)
    vMin = 0.0001
    vMax = 3.
    return bisect(smileMin, vMin, vMax, args=(S, K, T, price, r, call_put), rtol=1e-15, full_output=False, disp=True)


class qrHeston:
    """
    A Monte Carlo class for simulating the stochastic models (ex: Heston, rough Heston...)
            
     
    """
    def __init__(self, qrheston_params, S0, dt = 0.004, Tmin = 0.25, Tmax=2., r = 0.0):
        
        # Time discretisation parameters, maturitiy <= 2 year
        self.Tmax  = Tmax  
        self.Tmin = Tmin
        self.dt =  dt #Tmax/nbTimeSteps
        
        # Time gird is divided into 3 parts, about 1 day for T<3months(0.25year), 2d for T<=6months, 3d for T>6months
        self.time_grids = [np.arange(0., Tmin, dt), np.arange(0., 2*Tmin, 2*dt), np.arange(0., Tmax, 3*dt)]
        
        # Spot price
        self.S0 = S0
        
        # risk-free interest
        self.r = r
        
        # qrHeston parameters
        self.alpha = qrheston_params['alpha']
        self.Lambda = qrheston_params['Lambda']
        self.a = qrheston_params['a']
        self.b = qrheston_params['b']
        self.c = qrheston_params['c']
        self.Z0 = qrheston_params['Z0']
        
  
    def qrHeston_single_Paths(self, T, nbPaths):
        """
        Monte Carlo Simulation for the quadratic rough Heston model
        Input: 
            T: maturity
            nbPaths: number of paths
        
        Output:
            An array of Stock price paths at maturity T, with each row(axis=0) a simulation path 
            
        """
        ## choose the proper time grid for the maturity T
        if T <= self.Tmin:
            time_grid = self.time_grids[0] 
        elif T <= 2*self.Tmin:
            time_grid = self.time_grids[1]
        else:
            time_grid = self.time_grids[2]
        dt = time_grid[1] - time_grid[0]
        
        return qrHeston_single_stockPaths(time_grid, nbPaths, T, dt, self.S0, self.alpha, self.Lambda, self.a, self.b, self.c, self.Z0)
        
            
        
    def qrHeston_multiple_Paths(self, nbPaths):
        """
        Monte Carlo Simulation for the quadratic rough Heston model
        Input: 
            nbPaths: number of paths
        
        Output:
            A list of 3 arrays of Stock price paths at maturity 3 months, 6 months and 2 years, respectively.
            
        """
        Ts = [self.Tmin, 2*self.Tmin, self.Tmax]
        multi_paths = [self.qrHeston_single_Paths(T, nbPaths) for T in Ts]
        
        return multi_paths
    
    
    def compute_relative_errors(self, sample):
        """
        Compute relative errors using 95% confidence intervals
        Input: 
            the sample data
        Output:
            Average, standard deviation and maximum relative errors
        """
        avg = np.mean(sample)
        sig= np.std(sample)/np.sqrt(len(sample))
        
        f = np.array(list(filter(lambda x: x!=0 and x >= avg-1.96*sig and x <= avg+1.96*sig, sample)))
        relative_errors = np.abs(f - avg)/f
        
        return 100*np.mean(relative_errors), 100*np.std(relative_errors), 100*np.max(relative_errors)
            
            
    def qrHeston_CallPut(self, strikes, maturities, N = 100000, call_put = 1):
        """
        Compute the call/put option price with call_put = 1/-1 for "Call/Put"  for given strikes and maturities
        
        Output:
            A list with each element the call/put prices for the strike(s) and maturity(maturities)
            A list of (Average, standard deviation and maximum) relative errors of the Monte Carlo for each maturity-strike
        """
        dim = len(strikes)*len(maturities)
        
        callput_prices = np.zeros(dim)
        errors = np.zeros((3, dim))
        
        multi_paths = self.qrHeston_multiple_Paths(N)
        
        i = 0
        for T in maturities:
            if T <= self.Tmin:
                paths = multi_paths[0]
                dt = self.dt
            elif T <= 2*self.Tmin:
                paths = multi_paths[1]
                dt = 2*self.dt
            else:
                paths = multi_paths[2]
                dt = 3*self.dt
                
            stop = int(T/dt)-1 ##+1?
            stockPrice = paths[:, stop]
            
            for K in strikes:
                tmp   = np.maximum(call_put*(stockPrice - K), 0.0)
                callput_prices[i] = np.mean(tmp)*np.exp(-self.r * T)
#                 try:
#                     errors[:, i] = self.compute_relative_errors(tmp)
#                 except:
#                     pass
                i += 1
        return callput_prices               
#         return callput_prices, errors


