from multiprocessing import Pool
# from quadratic_rHeston import *
from quadratic_rHeston_torch import *

from functools import partial
from tqdm import tqdm
import time
import os

# os.chdir("Data")
df = np.loadtxt("Data/parameters_3.txt")


S0 = 100.
r = 0.03
# maturities = np.round(np.linspace(0.1, 2., 8), 2) # Time to expiry in years
maturities = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
strikes = np.array([80, 85, 90, 95, 97, 99, 100, 101, 103, 105, 110, 115, 120])
# maturities = np.array([0.06, 0.15, 0.31, 0.56])
# strikes = np.array([5.25, 5.50, 5.75, 6.00, 6.25])
strike_dim = len(strikes)
maturities_dim = len(maturities)



def single_job(qrheston_params):

 # compute the implied vol from hestonParams
    MC = qrHeston(qrheston_params, S0=S0, r=r)
    call_prices = MC.qrHeston_CallPut(strikes, maturities)
    
    try:
        tmp = impliedVols(S0, strikes, maturities, call_prices,r=r)
        return np.concatenate((qrheston_params, tmp))

    except:
        pass

    
    
def run_apply_async(func, argument_list, num_processes):
    pool = Pool(num_processes)
    
#     jobs = [pool.apply_async(func = func, args=(*argument,)) if isinstance(argument, tuple) 
#             else pool.apply_async(func=func, args=(argument,)) for argument in argument_list ]
    jobs = [pool.apply_async(func=func, args=(argument,)) for argument in argument_list ]
    pool.close()
    
    result_list_tqdm = []
    print(len(jobs))
    for job in tqdm(jobs):
        tmp = job.get()
        
        if tmp is not None:
            result_list_tqdm.append(tmp)
            np.savetxt("Data/impliedVols_test.txt", np.array(result_list_tqdm, dtype=float), fmt = '%.4f')
    return

    
def main():
    print("Start : %s" % time.ctime())
    time.sleep(20)
    print("End : %s" % time.ctime())
    num_processes = 3
    func = single_job
    
    run_apply_async(func, argument_list=df[100:120], num_processes=num_processes)
    #save txt file
#     np.savetxt("Data/impliedVols.txt", result_list, fmt = '%.4f')
    
    #Save csv file
#     columns = ['alpha', 'lambda','a', 'b', 'c', 'Z0']
#     for T in maturities:
#         for K in strikes:
#             columns.append("iv(T=%s,K=%s)"%(T,K))
            
#     df_result = pd.DataFrame(result_list, columns = columns)
#     df_result.to_csv("Data/result.csv")
    
if __name__ == '__main__':
    main()
    

