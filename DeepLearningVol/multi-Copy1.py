from multiprocessing import Pool
import time


# define your job function
def single_job(para):

 # do what your job with configration from para
     print(para['INDEX'])

     pass

if __name__ =='__main__':



    # prepare all job configration
    para_ls = list()
    for idx in range(3):
        d = dict()
        d['INDEX'] = idx

        para_ls.append(d)
        pass

    # run mulit process
    pool = Pool()  # num of process to run
    result = []  # job function can return results if any

    for pds in para_ls:
        result.append(pool.apply_async(single_job, args=(pds,)))
        pass

    pool.close()    #关闭进程池，使其不再接受新任务
    pool.join()     #主进程阻塞等待子进程的退出，join方法要在close或terminate之后使用

    # save results
    final_result = list()
    for rslt in result:
        r = rslt.get()
        print(r)
        final_result.append(r)
        pass

# def say(msg):
#     print('msg : %s' %msg)
#     time.sleep(3)
#     print('end')
    
    
#     print('Start excuting the main process...')
#     start_time = time.time()
#     pool = Pool(3)
#     print('Start excuting 3 sub-processes...')
#     for i in range(3):
#         pool.apply_async(say, [i])
#     pool.close()
#     pool.join()
#     print('Main process ended, spent time : %s' % (time.time() - start_time))