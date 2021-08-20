import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm
### Price of the Call/Put option with the Black Scholes model

def BlackScholesCallPut(S, K, T, sigma, r=0.0, q=0.0, call_put=1):
    d1 = (np.log(S/K) + (r-q+.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put*(S*np.exp(-q*T)*norm.cdf(call_put*d1) - K*np.exp (-r*T) * norm.cdf (call_put*d2))

#### Compute the implied volatility

def impliedVol(S, K, T, price, r=0.0, q=0.0, call_put=1):
    def smileMin(vol, *args):
        S, K, T, price, r, q, call_put = args
        return price - BlackScholesCallPut(S, K, T, vol, r, q, call_put)
    vMin = 0.0001
    vMax = 3.
    return bisect(smileMin, vMin, vMax, args=(S, K, T, price, r, q, call_put), rtol=1e-15, full_output=False, disp=True)

def impliedVols(S, strikes, maturities, prices, r=0.0, q=0.0, call_put=1):
    results = []
    maturities_dim = len(maturities)
    strikes_dim = len(strikes)
    
    return np.array([[impliedVol(S, strikes[j], maturities[i], prices[i*strikes_dim + j], r, q, call_put) for j in range(strikes_dim)] for i in range(maturities_dim)]).ravel()




def vega(S, K, T, sigma, r=0.0):
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + T*(r+ 0.5 * sigma ** 2)) / (sigma * sqrtT)
        d2 = d1 - sigma*sqrtT
        N_prime = np.exp(-d1**2/2)/np.sqrt(2*np.pi)
        
        return S*sqrtT*N_prime 
    
def impliedVol_Bisect(S, K, T, price, r=0.0, call_put=1):
    def smileMin(vol, *args):
        S, K, T, price, r, call_put = args
        return price - BlackScholesCallPut(S, K, T, vol, r, call_put)
    vMin = 0.0001
    vMax = 3.
    return bisect(smileMin, vMin, vMax, args=(S, K, T, price, r, call_put), rtol=1e-15, full_output=False, disp=True)

def impliedVol_Netwton(S, K, T, price, r=0.0, call_put=1):
    x0 = -1
    x1 = np.sqrt(2/T*abs(np.log(S*np.exp(r*T)/K)))
    tol = 1e-15
    
    while abs(x1-x0) > tol:
        x0 = x1
        BSprice = BlackScholesCallPut(S, K, T, x1, r, call_put)
        veg = vega(S, K, T, x1, r)
        x1 = x1 - (BSprice - price)/veg
    return x1
        