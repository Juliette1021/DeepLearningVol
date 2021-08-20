import numpy as np
import pandas as pd
import scipy.stats as sps
import math
from scipy.special import gamma
from scipy.optimize import bisect
from scipy.stats import norm

# from utils import *
import torch
torch.set_printoptions(precision=10)

class qrHeston:
    """
    A Monte Carlo class for simulating the stochastic models (ex: Heston, rough Heston...)
            
     
    """
    def __init__(self, params, S0, dt = 0.004, Tmin = 0.25, Tmax=1., r = 0.0):
        
        # Time discretisation parameters, maturitiy <= 2 year
        self.Tmax  = Tmax  
        self.Tmin = Tmin
        self.dt =  dt #Tmax/nbTimeSteps
        
        # Time gird is divided into 3 parts, about 1 day for T<3months(0.25year), 2d for T<=6months, 3d for T>6months
        self.time_grids = [torch.arange(0., Tmin, dt), torch.arange(0., 2*Tmin, 2*dt), torch.arange(0., Tmax, 3*dt)]
        
        # Spot price
        self.S0 = S0
        
        # risk-free interest
        self.r = r
        
        # qrHeston parameters
        self.alpha, self.Lambda, self.a, self.b, self.c, self.Z0 = params
        self.coef = self.Lambda/gamma(self.alpha)
        
    
    
    def qrHeston_single_stockPaths(self, T, nbPaths):
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
#         dtsqrt= torch.sqrt(dt)
        
        nbTimeSteps = int(T/dt)
    
        # Generate a Brownian Motion sequence
#         W = torch.randn([nbPaths, nbTimeSteps])*dtsqrt
        W = torch.normal(mean=torch.zeros([nbPaths, nbTimeSteps]), std=torch.sqrt(dt))
        X, Z, V = torch.zeros([3, nbPaths, nbTimeSteps])
       
        X[:, 0] = torch.full((nbPaths, ), np.log(self.S0))
        Z[:, 0] = torch.full((nbPaths, ), self.Z0)
        V[:, 0] = torch.full((nbPaths, ), self.a * (self.Z0 - self.b)**2 + self.c)

        for i in range(1, nbTimeSteps):
            ti   = torch.pow(time_grid[i] - time_grid[: i], self.alpha - 1)
#             print(ti)
            Zi   = Z[:, : i]
            Vi   = V[:, : i]
            Wi   = W[:, : i] 
            tmp = torch.matmul(dt*Zi - torch.sqrt(Vi)*Wi, ti)
#             print(tmp.shape)
            Z[:, i] = self.Z0 - self.coef*tmp
#             Z[i] = self.Z0 - coef*dt*np.sum(ti*Zi) + coef*np.sum(ti*np.sqrt(Vi)*Wi)
            V[:, i] = self.a * (Z[:, i] - self.b)**2 + self.c
            X[:, i] = X[:, i-1] -0.5*V[:, i-1]*dt + torch.sqrt(V[:, i-1])*W[:, i]
        
        return torch.exp(X)
    
    def qrHeston_multiple_stockPaths(self, nbPaths, T_max):
        """
        Monte Carlo Simulation for the quadratic rough Heston model
        Input: 
            nbPaths: number of paths
        
        Output:
            A list of 3 arrays of Stock price paths at maturity 3 months, 6 months and 2 years, respectively.
            
        """
        if T_max > 2*self.Tmin:
            Ts = [self.Tmin, 2*self.Tmin, self.Tmax]
        elif T_max > self.Tmin:
            Ts = [self.Tmin, 2*self.Tmin]
        else:
            Ts = [self.Tmin]
            
        multi_paths = [self.qrHeston_single_stockPaths(T, nbPaths) for T in Ts]
        
        return multi_paths
        
            
    def qrHeston_CallPut(self, strikes, maturities, N = 100000, call_put = 1):
        """
        Compute the call/put option price with call_put = 1/-1 for "Call/Put"  for given strikes and maturities
        
        Output:
            A list with each element the call/put prices for the strike(s) and maturity(maturities)
            A list of (Average, standard deviation and maximum) relative errors of the Monte Carlo for each maturity-strike
        """
        callput_prices = torch.zeros(len(strikes)*len(maturities))
#         errors = tf.zeros((3, dim))
        
        multi_paths = self.qrHeston_multiple_stockPaths(N, T_max=maturities[-1])
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
                tmp   = torch.maximum(call_put*(stockPrice - K), torch.tensor([0.0]))
                callput_prices[i] = torch.mean(tmp)*torch.exp(torch.tensor([-self.r * T]))
#                 try:
#                     errors[:, i] = self.compute_relative_errors(tmp)
#                 except:
#                     pass
                i += 1
        return callput_prices.numpy()           
#         return callput_prices, errors


def BlackScholesCallPut(S, K, T, sigma, r=0.0, call_put=1):
    d1 = (np.log(S / K) + (r + .5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put * (S * norm.cdf(call_put * d1) - K * np.exp(-r * T) * norm.cdf(call_put * d2))


#### Compute the implied volatility

def impliedVol(S, K, T, price, r=0.0, call_put=1):
    def smileMin(vol, *args):
        S, K, T, price, r, call_put = args
        return price - BlackScholesCallPut(S, K, T, vol, r, call_put)

    vMin = 0.0001
    vMax = 3.
    return bisect(smileMin, vMin, vMax, args=(S, K, T, price, r, call_put), rtol=1e-15, full_output=False, disp=True)


def impliedVols(S, strikes, maturities, prices, r=0.0, call_put=1):
    results = []
    maturities_dim = len(maturities)
    strikes_dim = len(strikes)

    return np.array([[impliedVol(S, strikes[j], maturities[i], prices[i * strikes_dim + j], r, call_put) for j in
                      range(strikes_dim)] for i in range(maturities_dim)]).ravel()