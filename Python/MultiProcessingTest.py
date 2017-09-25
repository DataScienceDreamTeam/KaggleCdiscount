import multiprocessing as mp
import time
import itertools
import bson

def process(data):
    return data["_id"]


if __name__ == '__main__':

    bson_filepath = "C:\\Users\\gael.superi\\Documents\\GitHub\\KaggleCdiscount\\Data\\train.bson"
    bson_file_iter = bson.decode_file_iter(open(bson_filepath, "rb"))

    pool = mp.Pool(mp.cpu_count() * 4)

    nbProducts = 0

    results = []
    t0 = time.time()
    for k in range(36):  

        pool = mp.Pool(mp.cpu_count() * 4)  
        print("processing, k = %i" % k)
        data_slice = itertools.islice(bson_file_iter, 200000)
        result = pool.map_async(process,data_slice)

        while not result.ready():
            print("wait for, k = %i" % k)
            result.wait(1000)

        results.append(result)


    t1 = time.time()
    total_time = t1 - t0
    print("Time for initializing pools : %s" % str(total_time))

    for res in results:    
        real_result = res.get()
        nbProducts = nbProducts + len(real_result)
        print(nbProducts)

    t2 = time.time()
    total_time = t2 - t1
    print("Time forget results: %s" % str(total_time))

    pool.close()
    pool.join()

    t3 = time.time()
    total_time = t3 - t2
    print("Time for close and join : %s" % str(total_time))

    print(nbProducts)




    # print("start")
    # pool=mp.Pool(2)
    # pool.map_async(Func,range(100))

    # print("texte 1")
    # print("texte 2")
    # print("texte 3")
    # print("end")

    # pool.close()
    # pool.join()